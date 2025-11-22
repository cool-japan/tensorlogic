//! TPTP Format Parser
//!
//! Parses logic expressions from TPTP (Thousands of Problems for Theorem Provers)
//! format into TensorLogic IR.
//!
//! ## Supported Syntax
//!
//! - **FOF formulas**: `fof(name, role, formula).`
//! - **CNF formulas**: `cnf(name, role, clause).`
//! - **Logical operators**: `&` (and), `|` (or), `~` (not), `=>` (implies)
//! - **Quantifiers**: `!` (forall), `?` (exists)
//!
//! ## Examples
//!
//! ```
//! use tensorlogic_compiler::import::tptp::parse_tptp;
//!
//! // FOF formula
//! let expr = parse_tptp("fof(mortality, axiom, ![X]: (human(X) => mortal(X))).").unwrap();
//! ```

use anyhow::{anyhow, Result};
use tensorlogic_ir::{TLExpr, Term};

/// Parse TPTP formula into TLExpr
///
/// # Arguments
///
/// * `input` - TPTP formula string
///
/// # Returns
///
/// Parsed `TLExpr` or error if parsing fails
pub fn parse_tptp(input: &str) -> Result<TLExpr> {
    let input = input.trim().trim_end_matches('.');

    // Extract formula from fof(...) or cnf(...)
    if input.starts_with("fof(") {
        parse_fof(input)
    } else if input.starts_with("cnf(") {
        parse_cnf(input)
    } else {
        // Direct formula (no wrapper)
        parse_tptp_formula(input)
    }
}

fn parse_fof(input: &str) -> Result<TLExpr> {
    // fof(name, role, formula)
    if !input.starts_with("fof(") || !input.ends_with(')') {
        return Err(anyhow!("Invalid fof syntax: {}", input));
    }

    let content = &input[4..input.len() - 1];
    let parts: Vec<&str> = split_top_level(content, ',');

    if parts.len() != 3 {
        return Err(anyhow!("fof requires 3 arguments: name, role, formula"));
    }

    let formula = parts[2].trim();
    parse_tptp_formula(formula)
}

fn parse_cnf(input: &str) -> Result<TLExpr> {
    // cnf(name, role, clause)
    if !input.starts_with("cnf(") || !input.ends_with(')') {
        return Err(anyhow!("Invalid cnf syntax: {}", input));
    }

    let content = &input[4..input.len() - 1];
    let parts: Vec<&str> = split_top_level(content, ',');

    if parts.len() != 3 {
        return Err(anyhow!("cnf requires 3 arguments: name, role, clause"));
    }

    let clause = parts[2].trim();
    parse_tptp_formula(clause)
}

fn parse_tptp_formula(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    // Remove outer parentheses if present
    let input = if input.starts_with('(') && input.ends_with(')') {
        &input[1..input.len() - 1]
    } else {
        input
    };

    // Handle quantifiers: ![X]: formula or ?[X]: formula
    if input.starts_with('!') {
        return parse_tptp_quantifier(input, false);
    }

    if input.starts_with('?') {
        return parse_tptp_quantifier(input, true);
    }

    // Handle binary operators (with precedence)
    // Precedence: => (lowest), | , & (highest)

    // Implication (=>)
    if let Some(pos) = find_binary_op(input, "=>") {
        let left = parse_tptp_formula(input[..pos].trim())?;
        let right = parse_tptp_formula(input[pos + 2..].trim())?;
        return Ok(TLExpr::imply(left, right));
    }

    // Disjunction (|)
    if let Some(pos) = find_binary_op(input, "|") {
        let left = parse_tptp_formula(input[..pos].trim())?;
        let right = parse_tptp_formula(input[pos + 1..].trim())?;
        return Ok(TLExpr::or(left, right));
    }

    // Conjunction (&)
    if let Some(pos) = find_binary_op(input, "&") {
        let left = parse_tptp_formula(input[..pos].trim())?;
        let right = parse_tptp_formula(input[pos + 1..].trim())?;
        return Ok(TLExpr::and(left, right));
    }

    // Negation (~)
    if let Some(stripped) = input.strip_prefix('~') {
        let inner = parse_tptp_formula(stripped.trim())?;
        return Ok(TLExpr::negate(inner));
    }

    // Atomic formula (predicate)
    parse_tptp_atom(input)
}

fn parse_tptp_quantifier(input: &str, is_exists: bool) -> Result<TLExpr> {
    // Format: ![X]: formula or ![X, Y]: formula
    // Or: ?[X]: formula
    let input = input.trim();

    if !input.starts_with(if is_exists { '?' } else { '!' }) {
        return Err(anyhow!("Invalid quantifier syntax"));
    }

    let input = &input[1..]; // Remove ! or ?

    if !input.starts_with('[') {
        return Err(anyhow!("Expected [ after quantifier"));
    }

    let bracket_end = input
        .find(']')
        .ok_or_else(|| anyhow!("Missing ] in quantifier"))?;
    let vars_str = &input[1..bracket_end];
    let rest = &input[bracket_end + 1..].trim();

    // Expect : after ]
    if !rest.starts_with(':') {
        return Err(anyhow!("Expected : after variable list"));
    }

    let body_str = rest[1..].trim();

    // Parse variables (may be comma-separated)
    let vars: Vec<&str> = vars_str.split(',').map(|s| s.trim()).collect();

    // For simplicity, assume all variables have domain "Entity"
    // (TPTP doesn't specify domains in basic syntax)
    let body = parse_tptp_formula(body_str)?;

    // Nest quantifiers for multiple variables
    let result = vars.iter().rev().fold(body, |acc, var| {
        if is_exists {
            TLExpr::exists(*var, "Entity", acc)
        } else {
            TLExpr::forall(*var, "Entity", acc)
        }
    });

    Ok(result)
}

fn parse_tptp_atom(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    // Parse predicate(args...)
    if let Some(open_paren) = input.find('(') {
        if !input.ends_with(')') {
            return Err(anyhow!("Unmatched parentheses in atom: {}", input));
        }

        let pred_name = input[..open_paren].trim();
        let args_str = &input[open_paren + 1..input.len() - 1];

        let args = parse_tptp_arguments(args_str)?;

        Ok(TLExpr::pred(pred_name, args))
    } else {
        // Atomic predicate (no arguments)
        Ok(TLExpr::pred(input, vec![]))
    }
}

fn parse_tptp_arguments(args_str: &str) -> Result<Vec<Term>> {
    if args_str.trim().is_empty() {
        return Ok(vec![]);
    }

    let parts = split_top_level(args_str, ',');
    parts
        .into_iter()
        .map(|arg| parse_tptp_term(arg.trim()))
        .collect()
}

fn parse_tptp_term(term_str: &str) -> Result<Term> {
    // Variable: uppercase first letter
    if term_str.chars().next().is_some_and(|c| c.is_uppercase()) {
        Ok(Term::var(term_str))
    } else {
        // Constant (numeric or symbolic)
        Ok(Term::constant(term_str))
    }
}

fn split_top_level(input: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut current_start = 0;
    let mut depth = 0;

    for (i, ch) in input.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            c if c == delim && depth == 0 => {
                parts.push(&input[current_start..i]);
                current_start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(&input[current_start..]);
    parts
}

fn find_binary_op(input: &str, op: &str) -> Option<usize> {
    let mut depth = 0;

    for (i, _) in input.char_indices() {
        let ch = input.chars().nth(i)?;

        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ if depth == 0 && input[i..].starts_with(op) => return Some(i),
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fof() {
        let expr = parse_tptp("fof(test, axiom, mortal(socrates)).").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_fof_with_implication() {
        let expr =
            parse_tptp("fof(mortality, axiom, human(socrates) => mortal(socrates)).").unwrap();
        assert!(matches!(expr, TLExpr::Imply { .. }));
    }

    #[test]
    fn test_fof_with_quantifier() {
        let expr = parse_tptp("fof(all_mortal, axiom, ![X]: (human(X) => mortal(X))).").unwrap();
        assert!(matches!(expr, TLExpr::ForAll { .. }));
    }

    #[test]
    fn test_exists_quantifier() {
        let expr = parse_tptp("fof(some_mortal, axiom, ?[X]: mortal(X)).").unwrap();
        assert!(matches!(expr, TLExpr::Exists { .. }));
    }

    #[test]
    fn test_conjunction() {
        let expr = parse_tptp("fof(test, axiom, human(X) & mortal(X)).").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_disjunction() {
        let expr = parse_tptp("fof(test, axiom, human(X) | god(X)).").unwrap();
        assert!(matches!(expr, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_negation() {
        let expr = parse_tptp("fof(test, axiom, ~mortal(X)).").unwrap();
        assert!(matches!(expr, TLExpr::Not(_)));
    }

    #[test]
    fn test_multiple_quantifiers() {
        let expr = parse_tptp("fof(test, axiom, ![X, Y]: knows(X, Y)).").unwrap();
        assert!(matches!(expr, TLExpr::ForAll { .. }));
    }

    #[test]
    fn test_direct_formula() {
        let expr = parse_tptp("human(X) => mortal(X)").unwrap();
        assert!(matches!(expr, TLExpr::Imply { .. }));
    }

    #[test]
    fn test_complex_nested() {
        let expr =
            parse_tptp("fof(test, axiom, ![X]: ((human(X) & ~god(X)) => mortal(X))).").unwrap();
        assert!(matches!(expr, TLExpr::ForAll { .. }));
    }
}
