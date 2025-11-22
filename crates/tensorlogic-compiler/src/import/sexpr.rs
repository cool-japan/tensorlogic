//! S-Expression Parser
//!
//! Parses logic expressions from Lisp-like S-expression syntax into TensorLogic IR.
//!
//! ## Supported Syntax
//!
//! - **Predicates**: `(P x y)`
//! - **Conjunction**: `(and expr1 expr2 ...)`
//! - **Disjunction**: `(or expr1 expr2 ...)`
//! - **Negation**: `(not expr)`
//! - **Implication**: `(=> premise conclusion)`
//! - **Existential**: `(exists (var domain) expr)`
//! - **Universal**: `(forall (var domain) expr)`
//!
//! ## Examples
//!
//! ```
//! use tensorlogic_compiler::import::sexpr::parse_sexpr;
//!
//! // Conjunction
//! let expr = parse_sexpr("(and (human socrates) (mortal socrates))").unwrap();
//!
//! // Universal quantification
//! let expr = parse_sexpr("(forall (x Person) (=> (human x) (mortal x)))").unwrap();
//! ```

use anyhow::{anyhow, Result};
use tensorlogic_ir::{TLExpr, Term};

/// Parse S-expression into TLExpr
///
/// # Arguments
///
/// * `input` - S-expression string
///
/// # Returns
///
/// Parsed `TLExpr` or error if parsing fails
pub fn parse_sexpr(input: &str) -> Result<TLExpr> {
    let tokens = tokenize(input)?;
    let (expr, _) = parse_expr(&tokens, 0)?;
    Ok(expr)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    Symbol(String),
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in input.chars() {
        match ch {
            '(' => {
                if !current.is_empty() {
                    tokens.push(Token::Symbol(current.clone()));
                    current.clear();
                }
                tokens.push(Token::LParen);
            }
            ')' => {
                if !current.is_empty() {
                    tokens.push(Token::Symbol(current.clone()));
                    current.clear();
                }
                tokens.push(Token::RParen);
            }
            ' ' | '\t' | '\n' | '\r' => {
                if !current.is_empty() {
                    tokens.push(Token::Symbol(current.clone()));
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        tokens.push(Token::Symbol(current));
    }

    Ok(tokens)
}

fn parse_expr(tokens: &[Token], pos: usize) -> Result<(TLExpr, usize)> {
    if pos >= tokens.len() {
        return Err(anyhow!("Unexpected end of tokens"));
    }

    match &tokens[pos] {
        Token::LParen => {
            // List form: (operator args...)
            if pos + 1 >= tokens.len() {
                return Err(anyhow!("Empty list not allowed"));
            }

            if let Token::Symbol(op) = &tokens[pos + 1] {
                match op.as_str() {
                    "and" => parse_binary_chain(tokens, pos + 2, TLExpr::and),
                    "or" => parse_binary_chain(tokens, pos + 2, TLExpr::or),
                    "not" => {
                        let (inner, next_pos) = parse_expr(tokens, pos + 2)?;
                        expect_rparen(tokens, next_pos)?;
                        Ok((TLExpr::negate(inner), next_pos + 1))
                    }
                    "=>" => {
                        let (premise, pos1) = parse_expr(tokens, pos + 2)?;
                        let (conclusion, pos2) = parse_expr(tokens, pos1)?;
                        expect_rparen(tokens, pos2)?;
                        Ok((TLExpr::imply(premise, conclusion), pos2 + 1))
                    }
                    "exists" => parse_quantifier(tokens, pos + 2, true),
                    "forall" => parse_quantifier(tokens, pos + 2, false),
                    _ => {
                        // Predicate application
                        parse_predicate(tokens, pos + 1)
                    }
                }
            } else {
                Err(anyhow!("Expected operator after ("))
            }
        }
        Token::Symbol(sym) => {
            // Atomic predicate (no arguments)
            Ok((TLExpr::pred(sym, vec![]), pos + 1))
        }
        Token::RParen => Err(anyhow!("Unexpected )")),
    }
}

fn parse_binary_chain<F>(tokens: &[Token], mut pos: usize, op: F) -> Result<(TLExpr, usize)>
where
    F: Fn(TLExpr, TLExpr) -> TLExpr,
{
    let (first, next_pos) = parse_expr(tokens, pos)?;
    pos = next_pos;

    let mut exprs = vec![first];

    loop {
        if pos >= tokens.len() {
            return Err(anyhow!("Unexpected end of tokens in chain"));
        }

        if let Token::RParen = tokens[pos] {
            break;
        }

        let (expr, next_pos) = parse_expr(tokens, pos)?;
        exprs.push(expr);
        pos = next_pos;
    }

    if exprs.is_empty() {
        return Err(anyhow!("Empty chain not allowed"));
    }

    let result = exprs.into_iter().reduce(op).unwrap();

    Ok((result, pos + 1))
}

fn parse_quantifier(tokens: &[Token], pos: usize, is_exists: bool) -> Result<(TLExpr, usize)> {
    // Expect (var domain)
    if pos >= tokens.len() || tokens[pos] != Token::LParen {
        return Err(anyhow!("Expected ( after quantifier"));
    }

    if pos + 1 >= tokens.len() {
        return Err(anyhow!("Expected variable name"));
    }

    let var = if let Token::Symbol(v) = &tokens[pos + 1] {
        v.clone()
    } else {
        return Err(anyhow!("Expected variable name"));
    };

    if pos + 2 >= tokens.len() {
        return Err(anyhow!("Expected domain name"));
    }

    let domain = if let Token::Symbol(d) = &tokens[pos + 2] {
        d.clone()
    } else {
        return Err(anyhow!("Expected domain name"));
    };

    if pos + 3 >= tokens.len() || tokens[pos + 3] != Token::RParen {
        return Err(anyhow!("Expected ) after domain"));
    }

    let (body, next_pos) = parse_expr(tokens, pos + 4)?;

    expect_rparen(tokens, next_pos)?;

    let result = if is_exists {
        TLExpr::exists(&var, &domain, body)
    } else {
        TLExpr::forall(&var, &domain, body)
    };

    Ok((result, next_pos + 1))
}

fn parse_predicate(tokens: &[Token], pos: usize) -> Result<(TLExpr, usize)> {
    if pos >= tokens.len() {
        return Err(anyhow!("Expected predicate name"));
    }

    let pred_name = if let Token::Symbol(name) = &tokens[pos] {
        name.clone()
    } else {
        return Err(anyhow!("Expected predicate name"));
    };

    let mut args = Vec::new();
    let mut current_pos = pos + 1;

    loop {
        if current_pos >= tokens.len() {
            return Err(anyhow!("Unexpected end of tokens"));
        }

        if let Token::RParen = tokens[current_pos] {
            break;
        }

        if let Token::Symbol(arg) = &tokens[current_pos] {
            args.push(parse_term_from_str(arg)?);
            current_pos += 1;
        } else {
            return Err(anyhow!("Expected term or )"));
        }
    }

    Ok((TLExpr::pred(&pred_name, args), current_pos + 1))
}

fn parse_term_from_str(s: &str) -> Result<Term> {
    // Variable: starts with uppercase
    if s.chars().next().is_some_and(|c| c.is_uppercase()) {
        Ok(Term::var(s))
    } else {
        // Constant (numeric or symbolic)
        Ok(Term::constant(s))
    }
}

fn expect_rparen(tokens: &[Token], pos: usize) -> Result<()> {
    if pos >= tokens.len() || tokens[pos] != Token::RParen {
        Err(anyhow!("Expected ) at position {}", pos))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_predicate() {
        let expr = parse_sexpr("(mortal socrates)").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_conjunction() {
        let expr = parse_sexpr("(and (human x) (mortal x))").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_disjunction() {
        let expr = parse_sexpr("(or (human x) (god x))").unwrap();
        assert!(matches!(expr, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_negation() {
        let expr = parse_sexpr("(not (mortal x))").unwrap();
        assert!(matches!(expr, TLExpr::Not(_)));
    }

    #[test]
    fn test_implication() {
        let expr = parse_sexpr("(=> (human x) (mortal x))").unwrap();
        assert!(matches!(expr, TLExpr::Imply { .. }));
    }

    #[test]
    fn test_exists() {
        let expr = parse_sexpr("(exists (x Person) (mortal x))").unwrap();
        assert!(matches!(expr, TLExpr::Exists { .. }));
    }

    #[test]
    fn test_forall() {
        let expr = parse_sexpr("(forall (x Person) (mortal x))").unwrap();
        assert!(matches!(expr, TLExpr::ForAll { .. }));
    }

    #[test]
    fn test_complex_expression() {
        let expr = parse_sexpr("(forall (x Person) (=> (and (human x) (not (god x))) (mortal x)))")
            .unwrap();
        assert!(matches!(expr, TLExpr::ForAll { .. }));
    }

    #[test]
    fn test_multiple_conjunction() {
        let expr = parse_sexpr("(and (P x) (Q x) (R x))").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_predicate_no_args() {
        let expr = parse_sexpr("alive").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }
}
