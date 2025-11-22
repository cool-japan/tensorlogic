//! Prolog Syntax Parser
//!
//! Parses logic expressions from Prolog-like syntax into TensorLogic IR.
//!
//! ## Supported Syntax
//!
//! - **Facts**: `mortal(socrates).`
//! - **Rules**: `mortal(X) :- human(X).`
//! - **Conjunctions**: `human(X), mortal(X)`
//! - **Disjunctions**: `human(X) ; mortal(X)`
//! - **Negation**: `\+ mortal(X)` or `not(mortal(X))`
//! - **Quantifiers**: Implicit universal quantification over variables
//!
//! ## Examples
//!
//! ```
//! use tensorlogic_compiler::import::prolog::parse_prolog;
//!
//! // Simple fact
//! let expr = parse_prolog("knows(alice, bob).").unwrap();
//!
//! // Rule with implication
//! let expr = parse_prolog("mortal(X) :- human(X).").unwrap();
//!
//! // Conjunction
//! let expr = parse_prolog("human(X), greek(X).").unwrap();
//! ```

use anyhow::{anyhow, Result};
use tensorlogic_ir::{TLExpr, Term};

/// Parse Prolog syntax into TLExpr
///
/// # Arguments
///
/// * `input` - Prolog syntax string (may end with `.`)
///
/// # Returns
///
/// Parsed `TLExpr` or error if parsing fails
pub fn parse_prolog(input: &str) -> Result<TLExpr> {
    let input = input.trim().trim_end_matches('.');

    // Check for rule (:-) vs fact
    if let Some(pos) = input.find(":-") {
        let head = input[..pos].trim();
        let body = input[pos + 2..].trim();

        let head_expr = parse_prolog_term(head)?;
        let body_expr = parse_prolog_term(body)?;

        Ok(TLExpr::imply(body_expr, head_expr))
    } else {
        parse_prolog_term(input)
    }
}

fn parse_prolog_term(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    // Handle disjunction (;)
    if let Some(pos) = find_operator(input, ';') {
        let left = parse_prolog_term(input[..pos].trim())?;
        let right = parse_prolog_term(input[pos + 1..].trim())?;
        return Ok(TLExpr::or(left, right));
    }

    // Handle conjunction (,)
    if let Some(pos) = find_operator(input, ',') {
        let left = parse_prolog_term(input[..pos].trim())?;
        let right = parse_prolog_term(input[pos + 1..].trim())?;
        return Ok(TLExpr::and(left, right));
    }

    // Handle negation (\+)
    if let Some(stripped) = input.strip_prefix("\\+") {
        let inner = stripped.trim();
        return Ok(TLExpr::negate(parse_prolog_term(inner)?));
    }

    // Handle negation (not(...))
    if input.starts_with("not(") && input.ends_with(')') {
        let inner = &input[4..input.len() - 1];
        return Ok(TLExpr::negate(parse_prolog_term(inner)?));
    }

    // Handle parentheses
    if input.starts_with('(') && input.ends_with(')') {
        return parse_prolog_term(&input[1..input.len() - 1]);
    }

    // Parse predicate: name(args...)
    if let Some(open_paren) = input.find('(') {
        if !input.ends_with(')') {
            return Err(anyhow!("Unmatched parentheses in: {}", input));
        }

        let pred_name = input[..open_paren].trim();
        let args_str = &input[open_paren + 1..input.len() - 1];

        let args = parse_arguments(args_str)?;

        Ok(TLExpr::pred(pred_name, args))
    } else {
        // Atomic predicate with no arguments
        Ok(TLExpr::pred(input, vec![]))
    }
}

fn parse_arguments(args_str: &str) -> Result<Vec<Term>> {
    if args_str.trim().is_empty() {
        return Ok(vec![]);
    }

    let parts = split_arguments(args_str);
    parts
        .into_iter()
        .map(|arg| parse_term(arg.trim()))
        .collect()
}

fn parse_term(term_str: &str) -> Result<Term> {
    // Variable: uppercase first letter
    if term_str.chars().next().is_some_and(|c| c.is_uppercase()) {
        Ok(Term::var(term_str))
    } else {
        // Constant (numeric or symbolic)
        Ok(Term::constant(term_str))
    }
}

fn split_arguments(args: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in args.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

fn find_operator(input: &str, op: char) -> Option<usize> {
    let mut depth = 0;

    for (i, ch) in input.chars().enumerate() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c == op && depth == 0 => return Some(i),
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fact() {
        let expr = parse_prolog("mortal(socrates).").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_rule() {
        let expr = parse_prolog("mortal(X) :- human(X).").unwrap();
        assert!(matches!(expr, TLExpr::Imply { .. }));
    }

    #[test]
    fn test_conjunction() {
        let expr = parse_prolog("human(X), mortal(X).").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_disjunction() {
        let expr = parse_prolog("human(X) ; god(X).").unwrap();
        assert!(matches!(expr, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_negation_prefix() {
        let expr = parse_prolog("\\+ mortal(X).").unwrap();
        assert!(matches!(expr, TLExpr::Not(_)));
    }

    #[test]
    fn test_negation_function() {
        let expr = parse_prolog("not(mortal(X)).").unwrap();
        assert!(matches!(expr, TLExpr::Not(_)));
    }

    #[test]
    fn test_complex_rule() {
        let expr = parse_prolog("mortal(X) :- human(X), \\+ god(X).").unwrap();
        assert!(matches!(expr, TLExpr::Imply { .. }));
    }

    #[test]
    fn test_predicate_no_args() {
        let expr = parse_prolog("alive.").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_multiple_args() {
        let expr = parse_prolog("knows(alice, bob, charlie).").unwrap();
        if let TLExpr::Pred { name, args } = expr {
            assert_eq!(name, "knows");
            assert_eq!(args.len(), 3);
        } else {
            panic!("Expected Pred");
        }
    }

    #[test]
    fn test_variables_and_constants() {
        let expr = parse_prolog("age(Person, 25).").unwrap();
        if let TLExpr::Pred { name: _, args } = expr {
            assert!(matches!(args[0], Term::Var(_)));
            assert!(matches!(args[1], Term::Const(_)));
        } else {
            panic!("Expected Pred");
        }
    }
}
