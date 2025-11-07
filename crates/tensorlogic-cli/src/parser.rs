//! Enhanced expression parser for TensorLogic CLI
//!
//! Supports:
//! - Basic predicates: pred(x, y)
//! - Logical operators: AND, OR, NOT, IMPLIES
//! - Quantifiers: EXISTS, FORALL
//! - Arithmetic: +, -, *, /
//! - Comparisons: =, <, >, <=, >=, !=
//! - Conditionals: IF-THEN-ELSE
//! - Parentheses for grouping

use anyhow::{bail, Result};
use tensorlogic_ir::{TLExpr, Term};

/// Parse expression string with enhanced syntax support
pub fn parse_expression(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    if input.is_empty() {
        bail!("Empty expression");
    }

    // Handle IF-THEN-ELSE first to avoid operator splitting
    if input.starts_with("IF ") || input.starts_with("if ") {
        return parse_conditional(input);
    }

    // Parse with operator precedence
    parse_implication(input)
}

// Operator precedence (lowest to highest):
// 1. IMPLIES (→)
// 2. OR (|, ||)
// 3. AND (&, &&)
// 4. Comparisons (=, <, >, <=, >=, !=)
// 5. Arithmetic (+, -)
// 6. Arithmetic (*, /)
// 7. NOT (~, !)
// 8. Quantifiers (EXISTS, FORALL)
// 9. Conditionals (IF-THEN-ELSE)
// 10. Predicates

fn parse_implication(input: &str) -> Result<TLExpr> {
    if let Some(pos) = find_operator(input, &["->", "IMPLIES", "=>", "→"]) {
        let left = parse_or(&input[..pos].trim())?;
        let right = parse_implication(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::imply(left, right));
    }
    parse_or(input)
}

fn parse_or(input: &str) -> Result<TLExpr> {
    if let Some(pos) = find_operator(input, &[" OR ", " | ", "||"]) {
        let left = parse_and(&input[..pos].trim())?;
        let right = parse_or(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Or(Box::new(left), Box::new(right)));
    }
    parse_and(input)
}

fn parse_and(input: &str) -> Result<TLExpr> {
    if let Some(pos) = find_operator(input, &[" AND ", " & ", "&&", "∧"]) {
        let left = parse_comparison(&input[..pos].trim())?;
        let right = parse_and(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::And(Box::new(left), Box::new(right)));
    }
    parse_comparison(input)
}

fn parse_comparison(input: &str) -> Result<TLExpr> {
    // Check for comparison operators
    if let Some(pos) = find_operator(input, &[" = ", " == "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Eq(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" <= ", " ≤ "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Lte(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" >= ", " ≥ "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Gte(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" < "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Lt(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" > "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Gt(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" != ", " ≠ "]) {
        let left = parse_additive(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + operator_len(&input[pos..])..].trim())?;
        let eq = TLExpr::Eq(Box::new(left), Box::new(right));
        return Ok(TLExpr::Not(Box::new(eq)));
    }

    parse_additive(input)
}

fn parse_additive(input: &str) -> Result<TLExpr> {
    if let Some(pos) = find_operator(input, &[" + "]) {
        let left = parse_multiplicative(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + 3..].trim())?;
        return Ok(TLExpr::Add(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" - "]) {
        let left = parse_multiplicative(&input[..pos].trim())?;
        let right = parse_additive(&input[pos + 3..].trim())?;
        return Ok(TLExpr::Sub(Box::new(left), Box::new(right)));
    }

    parse_multiplicative(input)
}

fn parse_multiplicative(input: &str) -> Result<TLExpr> {
    if let Some(pos) = find_operator(input, &[" * ", " × "]) {
        let left = parse_unary(&input[..pos].trim())?;
        let right = parse_multiplicative(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Mul(Box::new(left), Box::new(right)));
    }

    if let Some(pos) = find_operator(input, &[" / ", " ÷ "]) {
        let left = parse_unary(&input[..pos].trim())?;
        let right = parse_multiplicative(&input[pos + operator_len(&input[pos..])..].trim())?;
        return Ok(TLExpr::Div(Box::new(left), Box::new(right)));
    }

    parse_unary(input)
}

fn parse_unary(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    // Handle NOT
    for prefix in &["NOT ", "not ", "~", "!", "¬"] {
        if let Some(rest) = input.strip_prefix(prefix) {
            let body = parse_unary(rest.trim())?;
            return Ok(TLExpr::Not(Box::new(body)));
        }
    }

    parse_primary(input)
}

fn parse_primary(input: &str) -> Result<TLExpr> {
    let input = input.trim();

    // Handle quantifiers
    if let Some(rest) = input
        .strip_prefix("EXISTS ")
        .or_else(|| input.strip_prefix("exists "))
        .or_else(|| input.strip_prefix("∃ "))
    {
        return parse_quantifier(rest, true);
    }

    if let Some(rest) = input
        .strip_prefix("FORALL ")
        .or_else(|| input.strip_prefix("forall "))
        .or_else(|| input.strip_prefix("∀ "))
    {
        return parse_quantifier(rest, false);
    }

    // Handle parentheses
    if input.starts_with('(') && input.ends_with(')') {
        let inner = &input[1..input.len() - 1];
        if is_balanced(inner) {
            return parse_expression(inner);
        }
    }

    // Handle numeric constants
    if let Ok(value) = input.parse::<f64>() {
        return Ok(TLExpr::Constant(value));
    }

    // Handle predicates
    if let Some(paren_pos) = input.find('(') {
        if input.ends_with(')') {
            let name = input[..paren_pos].trim();
            let args_str = &input[paren_pos + 1..input.len() - 1];

            let args: Vec<Term> = if args_str.trim().is_empty() {
                vec![]
            } else {
                args_str
                    .split(',')
                    .map(|a| parse_term(a.trim()))
                    .collect::<Result<Vec<_>>>()?
            };

            return Ok(TLExpr::pred(name, args));
        }
    }

    // Single variable or constant
    Ok(TLExpr::pred(input, vec![]))
}

fn parse_quantifier(input: &str, is_exists: bool) -> Result<TLExpr> {
    // Format: "x IN Domain. body" or "x. body"
    let parts: Vec<&str> = input.splitn(2, '.').collect();
    if parts.len() != 2 {
        bail!(
            "Invalid quantifier syntax: expected '{} VAR [IN DOMAIN]. BODY'",
            if is_exists { "EXISTS" } else { "FORALL" }
        );
    }

    let var_part = parts[0].trim();
    let body = parse_expression(parts[1].trim())?;

    // Check for "IN Domain" syntax
    let (var, domain) = if let Some(in_pos) = var_part.find(" IN ") {
        let var = var_part[..in_pos].trim();
        let domain = var_part[in_pos + 4..].trim();
        (var, domain)
    } else if let Some(in_pos) = var_part.find(" in ") {
        let var = var_part[..in_pos].trim();
        let domain = var_part[in_pos + 4..].trim();
        (var, domain)
    } else {
        // Default domain "D"
        (var_part, "D")
    };

    if is_exists {
        Ok(TLExpr::exists(var, domain, body))
    } else {
        Ok(TLExpr::forall(var, domain, body))
    }
}

fn parse_conditional(input: &str) -> Result<TLExpr> {
    // Format: "IF cond THEN then_expr ELSE else_expr"
    let input = input
        .strip_prefix("IF ")
        .or_else(|| input.strip_prefix("if "))
        .unwrap();

    let then_pos = input
        .find(" THEN ")
        .or_else(|| input.find(" then "))
        .ok_or_else(|| anyhow::anyhow!("Missing THEN in IF-THEN-ELSE"))?;

    let else_pos = input
        .find(" ELSE ")
        .or_else(|| input.find(" else "))
        .ok_or_else(|| anyhow::anyhow!("Missing ELSE in IF-THEN-ELSE"))?;

    let cond = parse_expression(&input[..then_pos].trim())?;
    let then_expr = parse_expression(&input[then_pos + 6..else_pos].trim())?;
    let else_expr = parse_expression(&input[else_pos + 6..].trim())?;

    Ok(TLExpr::IfThenElse {
        condition: Box::new(cond),
        then_branch: Box::new(then_expr),
        else_branch: Box::new(else_expr),
    })
}

fn parse_term(input: &str) -> Result<Term> {
    let input = input.trim();

    // Check if it's a quoted string (constant)
    if input.starts_with('"') && input.ends_with('"') {
        Ok(Term::Const(input[1..input.len() - 1].to_string()))
    } else {
        // Variable
        Ok(Term::var(input))
    }
}

/// Find the position of an operator at the top level (not inside parentheses)
fn find_operator(input: &str, operators: &[&str]) -> Option<usize> {
    let mut depth = 0;
    let chars: Vec<char> = input.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }

        if depth == 0 {
            for op in operators {
                if input[i..].starts_with(op) {
                    return Some(i);
                }
            }
        }
    }

    None
}

/// Get the length of the operator at the given position
fn operator_len(input: &str) -> usize {
    let operators = vec![
        "->", "IMPLIES", "=>", "→", " OR ", " | ", "||", " AND ", " & ", "&&", "∧", " = ", " == ",
        " <= ", " ≥ ", " >= ", " ≥ ", " < ", " > ", " != ", " ≠ ", " + ", " - ", " * ", " × ",
        " / ", " ÷ ",
    ];

    for op in operators {
        if input.starts_with(op) {
            return op.len();
        }
    }

    1
}

/// Check if parentheses are balanced
fn is_balanced(input: &str) -> bool {
    let mut depth = 0;
    for ch in input.chars() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_predicate() {
        let expr = parse_expression("knows(x, y)").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_and_operation() {
        let expr = parse_expression("p(x) AND q(y)").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_arithmetic() {
        let expr = parse_expression("x + y").unwrap();
        assert!(matches!(expr, TLExpr::Add(_, _)));
    }

    #[test]
    fn test_comparison() {
        let expr = parse_expression("x < y").unwrap();
        assert!(matches!(expr, TLExpr::Lt(_, _)));
    }

    #[test]
    fn test_quantifier() {
        let expr = parse_expression("EXISTS x IN Person. knows(x, y)").unwrap();
        assert!(matches!(expr, TLExpr::Exists { .. }));
    }

    #[test]
    fn test_conditional() {
        let expr = parse_expression("IF x < 0 THEN 0 ELSE x").unwrap();
        assert!(matches!(expr, TLExpr::IfThenElse { .. }));
    }

    #[test]
    fn test_complex_expression() {
        let expr = parse_expression("(p(x) OR q(y)) AND r(z)").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }
}
