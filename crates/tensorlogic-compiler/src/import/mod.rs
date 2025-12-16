//! Logic Expression Import
//!
//! This module provides parsers for importing logic expressions from various
//! external logic frameworks and formats.
//!
//! ## Supported Formats
//!
//! - **Prolog**: Standard Prolog syntax with predicates, conjunctions, disjunctions
//! - **S-Expression**: Lisp-like S-expression format for nested logic
//! - **TPTP**: TPTP (Thousands of Problems for Theorem Provers) format
//!
//! ## Examples
//!
//! ```
//! use tensorlogic_compiler::import::prolog::parse_prolog;
//!
//! // Parse Prolog-style rule
//! let expr = parse_prolog("mortal(X) :- human(X).").unwrap();
//! ```
//!
//! ```no_run
//! use tensorlogic_compiler::import::sexpr::parse_sexpr;
//!
//! // Parse S-expression
//! let expr = parse_sexpr("(forall x (=> (human x) (mortal x)))").unwrap();
//! ```

pub mod prolog;
pub mod sexpr;
pub mod tptp;

pub use prolog::parse_prolog;
pub use sexpr::parse_sexpr;
pub use tptp::parse_tptp;

use anyhow::{anyhow, Result};

/// Auto-detect format and parse logic expression
///
/// This function attempts to auto-detect the format based on syntax patterns
/// and parse accordingly.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::import::parse_auto;
///
/// // Prolog format
/// let expr1 = parse_auto("mortal(socrates).").unwrap();
///
/// // S-expression format
/// let expr2 = parse_auto("(and (P x) (Q x))").unwrap();
/// ```
pub fn parse_auto(input: &str) -> Result<tensorlogic_ir::TLExpr> {
    let trimmed = input.trim();

    // Detect format based on syntax (check TPTP before S-expr to avoid false positives)
    if trimmed.starts_with("fof(") || trimmed.starts_with("cnf(") {
        // TPTP format
        parse_tptp(trimmed)
    } else if trimmed.starts_with('(') {
        // S-expression format
        parse_sexpr(trimmed)
    } else if trimmed.contains(":-") || trimmed.contains('.') {
        // Prolog format
        parse_prolog(trimmed)
    } else {
        Err(anyhow!(
            "Unable to auto-detect format for input: {}",
            trimmed
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::TLExpr;

    #[test]
    fn test_auto_detect_sexpr() {
        let expr = parse_auto("(and (P x) (Q x))").unwrap();
        assert!(matches!(expr, TLExpr::And(_, _)));
    }

    #[test]
    fn test_auto_detect_prolog() {
        let expr = parse_auto("mortal(X).").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_auto_detect_unknown() {
        let result = parse_auto("random text without structure");
        assert!(result.is_err());
    }
}
