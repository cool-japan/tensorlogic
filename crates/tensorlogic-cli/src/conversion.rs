//! Format conversion utilities

use anyhow::{Context, Result};
use std::fs;
use tensorlogic_ir::TLExpr;

use crate::cli::ConvertFormat;
use crate::parser;

/// Convert between different formats
pub fn convert(
    input: &str,
    from: ConvertFormat,
    to: ConvertFormat,
    pretty: bool,
) -> Result<String> {
    // Parse input
    let expr = read_input(input, from)?;

    // Convert to output format
    write_output(&expr, to, pretty)
}

fn read_input(input: &str, format: ConvertFormat) -> Result<TLExpr> {
    match format {
        ConvertFormat::Expr => parser::parse_expression(input),
        ConvertFormat::Json => {
            let content = if input == "-" {
                use std::io::Read;
                let mut buffer = String::new();
                std::io::stdin().read_to_string(&mut buffer)?;
                buffer
            } else if std::path::Path::new(input).exists() {
                fs::read_to_string(input).context("Failed to read input file")?
            } else {
                // Try to parse as JSON string directly
                input.to_string()
            };
            serde_json::from_str(&content).context("Failed to parse JSON")
        }
        ConvertFormat::Yaml => {
            let content = if std::path::Path::new(input).exists() {
                fs::read_to_string(input).context("Failed to read input file")?
            } else {
                // Try to parse as YAML string directly
                input.to_string()
            };
            serde_yaml::from_str(&content).context("Failed to parse YAML")
        }
    }
}

fn write_output(expr: &TLExpr, format: ConvertFormat, pretty: bool) -> Result<String> {
    match format {
        ConvertFormat::Expr => {
            let s = format_expression(expr, pretty);
            Ok(s)
        }
        ConvertFormat::Json => {
            if pretty {
                serde_json::to_string_pretty(expr).context("Failed to serialize to JSON")
            } else {
                serde_json::to_string(expr).context("Failed to serialize to JSON")
            }
        }
        ConvertFormat::Yaml => serde_yaml::to_string(expr).context("Failed to serialize to YAML"),
    }
}

/// Format an expression as a human-readable string
pub fn format_expression(expr: &TLExpr, pretty: bool) -> String {
    if pretty {
        format_expression_pretty(expr, 0)
    } else {
        format_expression_compact(expr)
    }
}

fn format_expression_compact(expr: &TLExpr) -> String {
    match expr {
        TLExpr::Pred { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let arg_strs: Vec<String> = args.iter().map(format_term).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
        }
        TLExpr::And(left, right) => {
            format!(
                "({} AND {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Or(left, right) => {
            format!(
                "({} OR {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Not(inner) => {
            format!("NOT {}", format_expression_compact(inner))
        }
        TLExpr::Imply(left, right) => {
            format!(
                "({} -> {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Exists { var, domain, body } => {
            format!(
                "EXISTS {} IN {}. {}",
                var,
                domain,
                format_expression_compact(body)
            )
        }
        TLExpr::ForAll { var, domain, body } => {
            format!(
                "FORALL {} IN {}. {}",
                var,
                domain,
                format_expression_compact(body)
            )
        }
        TLExpr::Eq(left, right) => {
            format!(
                "{} = {}",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Lt(left, right) => {
            format!(
                "{} < {}",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Gt(left, right) => {
            format!(
                "{} > {}",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Lte(left, right) => {
            format!(
                "{} <= {}",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Gte(left, right) => {
            format!(
                "{} >= {}",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Add(left, right) => {
            format!(
                "({} + {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Sub(left, right) => {
            format!(
                "({} - {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Mul(left, right) => {
            format!(
                "({} * {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::Div(left, right) => {
            format!(
                "({} / {})",
                format_expression_compact(left),
                format_expression_compact(right)
            )
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            format!(
                "IF {} THEN {} ELSE {}",
                format_expression_compact(condition),
                format_expression_compact(then_branch),
                format_expression_compact(else_branch)
            )
        }
        TLExpr::Constant(val) => format!("{}", val),
        TLExpr::Score(inner) => format!("SCORE({})", format_expression_compact(inner)),
        _ => format!("{:?}", expr), // Fallback for other variants
    }
}

fn format_expression_pretty(expr: &TLExpr, indent: usize) -> String {
    let indent_str = "  ".repeat(indent);

    match expr {
        TLExpr::Pred { name, args } => {
            if args.is_empty() {
                format!("{}{}", indent_str, name)
            } else {
                let arg_strs: Vec<String> = args.iter().map(format_term).collect();
                format!("{}{}({})", indent_str, name, arg_strs.join(", "))
            }
        }
        TLExpr::And(left, right) | TLExpr::Or(left, right) => {
            let op = match expr {
                TLExpr::And(_, _) => "AND",
                TLExpr::Or(_, _) => "OR",
                _ => unreachable!(),
            };

            // Check if subexpressions are simple
            if is_simple(left) && is_simple(right) {
                format!(
                    "{}{} {} {}",
                    indent_str,
                    format_expression_compact(left),
                    op,
                    format_expression_compact(right)
                )
            } else {
                format!(
                    "{}{}(\n{},\n{}\n{})",
                    indent_str,
                    op,
                    format_expression_pretty(left, indent + 1),
                    format_expression_pretty(right, indent + 1),
                    indent_str
                )
            }
        }
        TLExpr::Not(inner) => {
            if is_simple(inner) {
                format!("{}NOT {}", indent_str, format_expression_compact(inner))
            } else {
                format!(
                    "{}NOT(\n{}\n{})",
                    indent_str,
                    format_expression_pretty(inner, indent + 1),
                    indent_str
                )
            }
        }
        TLExpr::Imply(left, right) => {
            if is_simple(left) && is_simple(right) {
                format!(
                    "{}{} -> {}",
                    indent_str,
                    format_expression_compact(left),
                    format_expression_compact(right)
                )
            } else {
                format!(
                    "{}IMPLIES(\n{},\n{}\n{})",
                    indent_str,
                    format_expression_pretty(left, indent + 1),
                    format_expression_pretty(right, indent + 1),
                    indent_str
                )
            }
        }
        TLExpr::Exists { var, domain, body } | TLExpr::ForAll { var, domain, body } => {
            let quantifier = match expr {
                TLExpr::Exists { .. } => "EXISTS",
                TLExpr::ForAll { .. } => "FORALL",
                _ => unreachable!(),
            };

            let domain_str = format!(" IN {}", domain);

            if is_simple(body) {
                format!(
                    "{}{} {}{}. {}",
                    indent_str,
                    quantifier,
                    var,
                    domain_str,
                    format_expression_compact(body)
                )
            } else {
                format!(
                    "{}{} {}{}.\n{}",
                    indent_str,
                    quantifier,
                    var,
                    domain_str,
                    format_expression_pretty(body, indent + 1)
                )
            }
        }
        _ => format!("{}{}", indent_str, format_expression_compact(expr)),
    }
}

fn format_term(term: &tensorlogic_ir::Term) -> String {
    match term {
        tensorlogic_ir::Term::Var(name) => name.clone(),
        tensorlogic_ir::Term::Const(name) => name.clone(),
        tensorlogic_ir::Term::Typed { value, .. } => format_term(value),
    }
}

fn is_simple(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Pred { .. }
            | TLExpr::Eq(_, _)
            | TLExpr::Lt(_, _)
            | TLExpr::Gt(_, _)
            | TLExpr::Lte(_, _)
            | TLExpr::Gte(_, _)
            | TLExpr::Constant(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_format_simple_predicate() {
        let expr = TLExpr::Pred {
            name: "knows".to_string(),
            args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
        };

        let formatted = format_expression(&expr, false);
        assert_eq!(formatted, "knows(x, y)");
    }

    #[test]
    fn test_format_and() {
        let expr = TLExpr::And(
            Box::new(TLExpr::Pred {
                name: "p".to_string(),
                args: vec![],
            }),
            Box::new(TLExpr::Pred {
                name: "q".to_string(),
                args: vec![],
            }),
        );

        let formatted = format_expression(&expr, false);
        assert_eq!(formatted, "(p AND q)");
    }

    #[test]
    fn test_format_exists() {
        let expr = TLExpr::Exists {
            var: "x".to_string(),
            domain: "Person".to_string(),
            body: Box::new(TLExpr::Pred {
                name: "knows".to_string(),
                args: vec![Term::Var("x".to_string()), Term::Const("alice".to_string())],
            }),
        };

        let formatted = format_expression(&expr, false);
        assert_eq!(formatted, "EXISTS x IN Person. knows(x, alice)");
    }

    #[test]
    fn test_convert_json_to_yaml() {
        let json_input = r#"{"Pred":{"name":"test","args":[{"Var":"x"}]}}"#;
        let result = convert(json_input, ConvertFormat::Json, ConvertFormat::Yaml, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_expr_to_json() {
        let expr_input = "knows(x, y)";
        let result = convert(expr_input, ConvertFormat::Expr, ConvertFormat::Json, false);
        assert!(result.is_ok());
    }
}
