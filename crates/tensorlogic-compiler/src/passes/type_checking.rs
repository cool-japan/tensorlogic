//! Type checking pass using predicate signatures.

use std::collections::HashMap;

use anyhow::{bail, Result};
use tensorlogic_ir::{IrError, SignatureRegistry, TLExpr, Term, TypeAnnotation};

/// Type checking context with signature registry
pub struct TypeChecker {
    registry: SignatureRegistry,
}

impl TypeChecker {
    pub fn new(registry: SignatureRegistry) -> Self {
        TypeChecker { registry }
    }

    /// Check that an expression is well-typed
    pub fn check_expr(&self, expr: &TLExpr) -> Result<()> {
        match expr {
            TLExpr::Pred { name, args } => {
                self.check_predicate(name, args)?;
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
                self.check_expr(left)?;
                self.check_expr(right)?;
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
                self.check_expr(inner)?;
            }
            TLExpr::Let {
                var: _,
                value,
                body,
            } => {
                self.check_expr(value)?;
                self.check_expr(body)?;
            }
            TLExpr::Exists {
                var: _,
                domain,
                body,
            }
            | TLExpr::ForAll {
                var: _,
                domain,
                body,
            }
            | TLExpr::SoftExists {
                var: _,
                domain,
                body,
                ..
            }
            | TLExpr::SoftForAll {
                var: _,
                domain,
                body,
                ..
            }
            | TLExpr::Aggregate {
                var: _,
                domain,
                body,
                ..
            } => {
                // Check that domain exists (would need context for this)
                // For now, just check the body
                let _ = domain; // Placeholder
                self.check_expr(body)?;
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expr(condition)?;
                self.check_expr(then_branch)?;
                self.check_expr(else_branch)?;
            }

            // Modal/temporal logic operators - not yet implemented, pass through with recursion
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => {
                self.check_expr(inner)?;
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
                self.check_expr(before)?;
                self.check_expr(after)?;
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_weight, alt_expr) in alternatives {
                    self.check_expr(alt_expr)?;
                }
            }

            TLExpr::Constant(_) => {
                // Constants are always well-typed
            }
        }
        Ok(())
    }

    /// Check a predicate application against its signature
    fn check_predicate(&self, name: &str, args: &[Term]) -> Result<()> {
        // Look up signature
        let signature = match self.registry.get(name) {
            Some(sig) => sig,
            None => {
                // No signature registered - skip type checking
                return Ok(());
            }
        };

        // Check arity
        if !signature.matches_arity(args.len()) {
            return Err(IrError::ArityMismatch {
                name: name.to_string(),
                expected: signature.arity,
                actual: args.len(),
            }
            .into());
        }

        // Check types
        let arg_types: Vec<Option<&TypeAnnotation>> = args.iter().map(|t| t.get_type()).collect();

        if !signature.matches_types(&arg_types) {
            // Find the specific mismatch for a better error message
            for (i, expected_type) in signature.arg_types.iter().enumerate() {
                if let Some(actual_type) = arg_types[i] {
                    if expected_type != actual_type {
                        return Err(IrError::TypeMismatch {
                            name: name.to_string(),
                            arg_index: i,
                            expected: expected_type.type_name.clone(),
                            actual: actual_type.type_name.clone(),
                        }
                        .into());
                    }
                }
            }
            // Generic type mismatch if we can't find specific one
            bail!("Type mismatch for predicate '{}'", name);
        }

        Ok(())
    }
}

/// Infer types from predicate applications
pub fn infer_types(
    expr: &TLExpr,
    registry: &SignatureRegistry,
) -> Result<HashMap<String, TypeAnnotation>> {
    let mut inferred_types = HashMap::new();

    infer_types_recursive(expr, registry, &mut inferred_types)?;

    Ok(inferred_types)
}

fn infer_types_recursive(
    expr: &TLExpr,
    registry: &SignatureRegistry,
    inferred_types: &mut HashMap<String, TypeAnnotation>,
) -> Result<()> {
    match expr {
        TLExpr::Pred { name, args } => {
            if let Some(signature) = registry.get(name) {
                for (i, arg) in args.iter().enumerate() {
                    if let Term::Var(var_name) = arg.untyped() {
                        if i < signature.arg_types.len() {
                            let expected_type = &signature.arg_types[i];
                            if let Some(existing_type) = inferred_types.get(var_name) {
                                if existing_type != expected_type {
                                    return Err(IrError::InconsistentTypes {
                                        var: var_name.clone(),
                                        type1: existing_type.type_name.clone(),
                                        type2: expected_type.type_name.clone(),
                                    }
                                    .into());
                                }
                            } else {
                                inferred_types.insert(var_name.clone(), expected_type.clone());
                            }
                        }
                    }
                }
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
            infer_types_recursive(left, registry, inferred_types)?;
            infer_types_recursive(right, registry, inferred_types)?;
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
            infer_types_recursive(inner, registry, inferred_types)?;
        }
        TLExpr::Let {
            var: _,
            value,
            body,
        } => {
            infer_types_recursive(value, registry, inferred_types)?;
            infer_types_recursive(body, registry, inferred_types)?;
        }
        TLExpr::Exists {
            var: _,
            domain: _,
            body,
        }
        | TLExpr::ForAll {
            var: _,
            domain: _,
            body,
        }
        | TLExpr::SoftExists {
            var: _,
            domain: _,
            body,
            ..
        }
        | TLExpr::SoftForAll {
            var: _,
            domain: _,
            body,
            ..
        }
        | TLExpr::Aggregate {
            var: _,
            domain: _,
            body,
            ..
        } => {
            infer_types_recursive(body, registry, inferred_types)?;
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            infer_types_recursive(condition, registry, inferred_types)?;
            infer_types_recursive(then_branch, registry, inferred_types)?;
            infer_types_recursive(else_branch, registry, inferred_types)?;
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => {
            infer_types_recursive(inner, registry, inferred_types)?;
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
            infer_types_recursive(before, registry, inferred_types)?;
            infer_types_recursive(after, registry, inferred_types)?;
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_weight, alt_expr) in alternatives {
                infer_types_recursive(alt_expr, registry, inferred_types)?;
            }
        }

        TLExpr::Constant(_) => {
            // Constants have no variables to infer
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::PredicateSignature;

    fn create_test_registry() -> SignatureRegistry {
        let mut registry = SignatureRegistry::new();

        registry.register(PredicateSignature::new(
            "knows",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Person")],
        ));

        registry.register(PredicateSignature::new(
            "likes",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Thing")],
        ));

        registry
    }

    #[test]
    fn test_type_checking_success() {
        let registry = create_test_registry();
        let checker = TypeChecker::new(registry);

        let expr = TLExpr::pred(
            "knows",
            vec![
                Term::typed_var("alice", "Person"),
                Term::typed_var("bob", "Person"),
            ],
        );

        assert!(checker.check_expr(&expr).is_ok());
    }

    #[test]
    fn test_type_checking_arity_mismatch() {
        let registry = create_test_registry();
        let checker = TypeChecker::new(registry);

        let expr = TLExpr::pred("knows", vec![Term::typed_var("alice", "Person")]);

        assert!(checker.check_expr(&expr).is_err());
    }

    #[test]
    fn test_type_checking_type_mismatch() {
        let registry = create_test_registry();
        let checker = TypeChecker::new(registry);

        let expr = TLExpr::pred(
            "knows",
            vec![
                Term::typed_var("alice", "Person"),
                Term::typed_var("book", "Thing"), // Should be Person
            ],
        );

        let result = checker.check_expr(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_checking_untyped_args() {
        let registry = create_test_registry();
        let checker = TypeChecker::new(registry);

        // Untyped arguments should be accepted
        let expr = TLExpr::pred("knows", vec![Term::var("alice"), Term::var("bob")]);

        assert!(checker.check_expr(&expr).is_ok());
    }

    #[test]
    fn test_type_checking_no_signature() {
        let registry = SignatureRegistry::new();
        let checker = TypeChecker::new(registry);

        // Predicate without signature should be accepted
        let expr = TLExpr::pred("unknown_pred", vec![Term::var("x")]);

        assert!(checker.check_expr(&expr).is_ok());
    }

    #[test]
    fn test_type_inference_basic() {
        let registry = create_test_registry();

        let expr = TLExpr::pred("knows", vec![Term::var("alice"), Term::var("bob")]);

        let inferred = infer_types(&expr, &registry).unwrap();

        assert_eq!(inferred.len(), 2);
        assert_eq!(inferred["alice"].type_name, "Person");
        assert_eq!(inferred["bob"].type_name, "Person");
    }

    #[test]
    fn test_type_inference_multiple_predicates() {
        let registry = create_test_registry();

        let expr = TLExpr::and(
            TLExpr::pred("knows", vec![Term::var("alice"), Term::var("bob")]),
            TLExpr::pred("likes", vec![Term::var("alice"), Term::var("book")]),
        );

        let inferred = infer_types(&expr, &registry).unwrap();

        assert_eq!(inferred.len(), 3);
        assert_eq!(inferred["alice"].type_name, "Person");
        assert_eq!(inferred["bob"].type_name, "Person");
        assert_eq!(inferred["book"].type_name, "Thing");
    }

    #[test]
    fn test_type_inference_conflict() {
        let registry = create_test_registry();

        // alice appears in both predicates with different expected types
        let expr = TLExpr::and(
            TLExpr::pred("knows", vec![Term::var("alice"), Term::var("bob")]), // alice: Person
            TLExpr::pred("likes", vec![Term::var("bob"), Term::var("alice")]), // alice: Thing
        );

        let result = infer_types(&expr, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_expressions() {
        let registry = create_test_registry();
        let checker = TypeChecker::new(registry);

        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::and(
                TLExpr::pred(
                    "knows",
                    vec![
                        Term::typed_var("x", "Person"),
                        Term::typed_var("bob", "Person"),
                    ],
                ),
                TLExpr::pred(
                    "likes",
                    vec![
                        Term::typed_var("x", "Person"),
                        Term::typed_var("book", "Thing"),
                    ],
                ),
            ),
        );

        assert!(checker.check_expr(&expr).is_ok());
    }
}
