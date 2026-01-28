//! Variable scope analysis pass.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use tensorlogic_ir::{IrError, TLExpr, Term, TypeAnnotation};

/// Scope information for a variable
#[derive(Debug, Clone)]
pub struct VariableScope {
    pub name: String,
    pub bound_in: ScopeType,
    pub type_annotation: Option<TypeAnnotation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeType {
    Quantifier { quantifier_type: String },
    Free,
}

/// Result of scope analysis
#[derive(Debug, Clone, Default)]
pub struct ScopeAnalysisResult {
    pub variables: HashMap<String, VariableScope>,
    pub unbound_variables: Vec<String>,
    pub type_conflicts: Vec<TypeConflict>,
}

#[derive(Debug, Clone)]
pub struct TypeConflict {
    pub variable: String,
    pub type1: String,
    pub type2: String,
}

/// Analyze variable scopes in an expression
pub fn analyze_scopes(expr: &TLExpr) -> Result<ScopeAnalysisResult> {
    let mut result = ScopeAnalysisResult::default();
    let mut bound_vars = HashSet::new();

    analyze_expr(expr, &mut bound_vars, &mut result)?;

    Ok(result)
}

fn analyze_expr(
    expr: &TLExpr,
    bound_vars: &mut HashSet<String>,
    result: &mut ScopeAnalysisResult,
) -> Result<()> {
    match expr {
        #[allow(unreachable_patterns)]
        TLExpr::Pred { name: _, args } => {
            // Check all variables in predicate arguments
            for term in args {
                check_term(term, bound_vars, result);
            }
        }
        TLExpr::And(left, right)
        | TLExpr::Or(left, right)
        | TLExpr::Imply(left, right)
        | TLExpr::Add(left, right)
        | TLExpr::Sub(left, right)
        | TLExpr::Mul(left, right)
        | TLExpr::Div(left, right)
        | TLExpr::Pow(left, right)
        | TLExpr::Mod(left, right)
        | TLExpr::Min(left, right)
        | TLExpr::Max(left, right)
        | TLExpr::Eq(left, right)
        | TLExpr::Lt(left, right)
        | TLExpr::Gt(left, right)
        | TLExpr::Lte(left, right)
        | TLExpr::Gte(left, right)
        | TLExpr::TNorm { left, right, .. }
        | TLExpr::TCoNorm { left, right, .. }
        | TLExpr::FuzzyImplication {
            premise: left,
            conclusion: right,
            ..
        } => {
            analyze_expr(left, bound_vars, result)?;
            analyze_expr(right, bound_vars, result)?;
        }
        TLExpr::Not(inner)
        | TLExpr::Score(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::FuzzyNot { expr: inner, .. }
        | TLExpr::WeightedRule { rule: inner, .. } => {
            analyze_expr(inner, bound_vars, result)?;
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            analyze_expr(condition, bound_vars, result)?;
            analyze_expr(then_branch, bound_vars, result)?;
            analyze_expr(else_branch, bound_vars, result)?;
        }
        TLExpr::Constant(_) => {
            // Constants have no variables to analyze
        }
        TLExpr::Exists {
            var,
            domain: _,
            body,
        }
        | TLExpr::ForAll {
            var,
            domain: _,
            body,
        }
        | TLExpr::SoftExists {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::SoftForAll {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::Aggregate {
            var,
            domain: _,
            body,
            ..
        } => {
            // Variable is bound in this scope
            let was_bound = bound_vars.contains(var);
            bound_vars.insert(var.clone());

            // Record the binding
            if !result.variables.contains_key(var) {
                result.variables.insert(
                    var.clone(),
                    VariableScope {
                        name: var.clone(),
                        bound_in: ScopeType::Quantifier {
                            quantifier_type: match expr {
                                TLExpr::Exists { .. } => "exists".to_string(),
                                TLExpr::ForAll { .. } => "forall".to_string(),
                                TLExpr::SoftExists { .. } => "soft_exists".to_string(),
                                TLExpr::SoftForAll { .. } => "soft_forall".to_string(),
                                TLExpr::Aggregate { .. } => "aggregate".to_string(),
                                _ => unreachable!(),
                            },
                        },
                        type_annotation: None,
                    },
                );
            }

            // Analyze the body
            analyze_expr(body, bound_vars, result)?;

            // Unbind if it wasn't previously bound
            if !was_bound {
                bound_vars.remove(var);
            }
        }
        TLExpr::Let { var, value, body } => {
            // Analyze value expression first (without the new variable bound)
            analyze_expr(value, bound_vars, result)?;
            // Then analyze body with the variable bound
            let was_bound = bound_vars.contains(var);
            bound_vars.insert(var.clone());
            analyze_expr(body, bound_vars, result)?;
            if !was_bound {
                bound_vars.remove(var);
            }
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => {
            analyze_expr(inner, bound_vars, result)?;
        }
        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            analyze_expr(before, bound_vars, result)?;
            analyze_expr(after, bound_vars, result)?;
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_weight, alt_expr) in alternatives {
                analyze_expr(alt_expr, bound_vars, result)?;
            }
        }
        // Counting quantifiers
        TLExpr::CountingExists {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::CountingForAll {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::ExactCount {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::Majority {
            var,
            domain: _,
            body,
        } => {
            // Variable is bound in this scope
            let was_bound = bound_vars.contains(var);
            bound_vars.insert(var.clone());

            // Record the binding
            if !result.variables.contains_key(var) {
                result.variables.insert(
                    var.clone(),
                    VariableScope {
                        name: var.clone(),
                        bound_in: ScopeType::Quantifier {
                            quantifier_type: match expr {
                                TLExpr::CountingExists { .. } => "counting_exists".to_string(),
                                TLExpr::CountingForAll { .. } => "counting_forall".to_string(),
                                TLExpr::ExactCount { .. } => "exact_count".to_string(),
                                TLExpr::Majority { .. } => "majority".to_string(),
                                _ => unreachable!(),
                            },
                        },
                        type_annotation: None,
                    },
                );
            }

            // Analyze the body
            analyze_expr(body, bound_vars, result)?;

            // Unbind if it wasn't previously bound
            if !was_bound {
                bound_vars.remove(var);
            }
        }
        // All other expression types (enhancements) - skip for now
        _ => {
            // For unimplemented expression types, no scope analysis yet
        }
    }

    Ok(())
}

fn check_term(term: &Term, bound_vars: &HashSet<String>, result: &mut ScopeAnalysisResult) {
    match term {
        Term::Var(var_name) => {
            if !bound_vars.contains(var_name) && !result.variables.contains_key(var_name) {
                // This is a free variable
                result.variables.insert(
                    var_name.clone(),
                    VariableScope {
                        name: var_name.clone(),
                        bound_in: ScopeType::Free,
                        type_annotation: None,
                    },
                );
                result.unbound_variables.push(var_name.clone());
            }

            // Check for type annotation
            if let Some(type_ann) = term.get_type() {
                if let Some(existing_scope) = result.variables.get_mut(var_name) {
                    if let Some(ref existing_type) = existing_scope.type_annotation {
                        if existing_type != type_ann {
                            result.type_conflicts.push(TypeConflict {
                                variable: var_name.clone(),
                                type1: existing_type.type_name.clone(),
                                type2: type_ann.type_name.clone(),
                            });
                        }
                    } else {
                        existing_scope.type_annotation = Some(type_ann.clone());
                    }
                }
            }
        }
        Term::Typed {
            value,
            type_annotation,
        } => {
            // Check the underlying term
            check_term(value, bound_vars, result);

            // Record type annotation
            if let Term::Var(var_name) = value.untyped() {
                if let Some(existing_scope) = result.variables.get_mut(var_name) {
                    if let Some(ref existing_type) = existing_scope.type_annotation {
                        if existing_type != type_annotation {
                            result.type_conflicts.push(TypeConflict {
                                variable: var_name.clone(),
                                type1: existing_type.type_name.clone(),
                                type2: type_annotation.type_name.clone(),
                            });
                        }
                    } else {
                        existing_scope.type_annotation = Some(type_annotation.clone());
                    }
                }
            }
        }
        Term::Const(_) => {
            // Constants don't need scope checking
        }
    }
}

/// Validate that all variables are properly bound
pub fn validate_scopes(expr: &TLExpr) -> Result<()> {
    let result = analyze_scopes(expr)?;

    if !result.unbound_variables.is_empty() {
        bail!(
            "Unbound variables found: {}",
            result.unbound_variables.join(", ")
        );
    }

    if !result.type_conflicts.is_empty() {
        let conflict = &result.type_conflicts[0];
        return Err(IrError::InconsistentTypes {
            var: conflict.variable.clone(),
            type1: conflict.type1.clone(),
            type2: conflict.type2.clone(),
        }
        .into());
    }

    Ok(())
}

/// Suggest quantifiers for unbound variables
pub fn suggest_quantifiers(expr: &TLExpr) -> Result<Vec<String>> {
    let result = analyze_scopes(expr)?;
    let mut suggestions = Vec::new();

    for unbound_var in &result.unbound_variables {
        suggestions.push(format!(
            "Consider adding a universal quantifier: ∀{}. <expr>",
            unbound_var
        ));
    }

    Ok(suggestions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_variable() {
        let expr = TLExpr::exists("x", "Domain", TLExpr::pred("p", vec![Term::var("x")]));

        let result = analyze_scopes(&expr).unwrap();
        assert!(result.unbound_variables.is_empty());
        assert_eq!(result.variables.len(), 1);
        assert_eq!(result.variables["x"].name, "x");
    }

    #[test]
    fn test_unbound_variable() {
        let expr = TLExpr::pred("p", vec![Term::var("x")]);

        let result = analyze_scopes(&expr).unwrap();
        assert_eq!(result.unbound_variables.len(), 1);
        assert_eq!(result.unbound_variables[0], "x");
    }

    #[test]
    fn test_mixed_bound_unbound() {
        // ∃x. p(x, y) - x is bound, y is free
        let expr = TLExpr::exists(
            "x",
            "Domain",
            TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]),
        );

        let result = analyze_scopes(&expr).unwrap();
        assert_eq!(result.unbound_variables.len(), 1);
        assert_eq!(result.unbound_variables[0], "y");
        assert_eq!(result.variables.len(), 2);
    }

    #[test]
    fn test_nested_quantifiers() {
        // ∃x. ∀y. p(x, y, z) - x and y are bound, z is free
        let expr = TLExpr::exists(
            "x",
            "Domain",
            TLExpr::forall(
                "y",
                "Domain",
                TLExpr::pred("p", vec![Term::var("x"), Term::var("y"), Term::var("z")]),
            ),
        );

        let result = analyze_scopes(&expr).unwrap();
        assert_eq!(result.unbound_variables.len(), 1);
        assert_eq!(result.unbound_variables[0], "z");
    }

    #[test]
    fn test_validate_scopes_success() {
        let expr = TLExpr::exists("x", "Domain", TLExpr::pred("p", vec![Term::var("x")]));

        assert!(validate_scopes(&expr).is_ok());
    }

    #[test]
    fn test_validate_scopes_failure() {
        let expr = TLExpr::pred("p", vec![Term::var("x")]);

        assert!(validate_scopes(&expr).is_err());
    }

    #[test]
    fn test_type_annotations() {
        let expr = TLExpr::pred(
            "p",
            vec![
                Term::typed_var("x", "Person"),
                Term::typed_var("x", "Person"), // Same type, OK
            ],
        );

        let result = analyze_scopes(&expr).unwrap();
        assert!(result.type_conflicts.is_empty());
    }

    #[test]
    fn test_type_conflicts() {
        let expr = TLExpr::pred(
            "p",
            vec![
                Term::typed_var("x", "Person"),
                Term::typed_var("x", "Thing"), // Different type, conflict!
            ],
        );

        let result = analyze_scopes(&expr).unwrap();
        assert_eq!(result.type_conflicts.len(), 1);
        assert_eq!(result.type_conflicts[0].variable, "x");
    }

    #[test]
    fn test_suggest_quantifiers() {
        let expr = TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]);

        let suggestions = suggest_quantifiers(&expr).unwrap();
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions[0].contains("x"));
        assert!(suggestions[1].contains("y"));
    }
}
