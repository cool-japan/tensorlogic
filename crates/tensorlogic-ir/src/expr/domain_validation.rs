//! Domain validation for expressions.

use std::collections::HashMap;

use crate::domain::DomainRegistry;
use crate::error::IrError;
use crate::expr::TLExpr;

impl TLExpr {
    /// Validate all domain references in this expression.
    ///
    /// Checks that:
    /// - All quantifier domains exist in the registry
    /// - Variables used consistently across different quantifiers
    pub fn validate_domains(&self, registry: &DomainRegistry) -> Result<(), IrError> {
        let mut var_domains = HashMap::new();
        self.collect_and_validate_domains(registry, &mut var_domains)
    }

    fn collect_and_validate_domains(
        &self,
        registry: &DomainRegistry,
        var_domains: &mut HashMap<String, String>,
    ) -> Result<(), IrError> {
        match self {
            TLExpr::Exists { var, domain, body }
            | TLExpr::ForAll { var, domain, body }
            | TLExpr::SoftExists {
                var, domain, body, ..
            }
            | TLExpr::SoftForAll {
                var, domain, body, ..
            } => {
                // Check domain exists
                registry.validate_domain(domain)?;

                // Check for consistent variable usage
                if let Some(existing_domain) = var_domains.get(var) {
                    if existing_domain != domain {
                        // Check if domains are compatible
                        if !registry.are_compatible(existing_domain, domain)? {
                            return Err(IrError::VariableDomainMismatch {
                                var: var.clone(),
                                expected: existing_domain.clone(),
                                actual: domain.clone(),
                            });
                        }
                    }
                } else {
                    var_domains.insert(var.clone(), domain.clone());
                }

                body.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Aggregate {
                var, domain, body, ..
            } => {
                // Check domain exists
                registry.validate_domain(domain)?;

                // Check for consistent variable usage
                if let Some(existing_domain) = var_domains.get(var) {
                    if existing_domain != domain {
                        // Check if domains are compatible
                        if !registry.are_compatible(existing_domain, domain)? {
                            return Err(IrError::VariableDomainMismatch {
                                var: var.clone(),
                                expected: existing_domain.clone(),
                                actual: domain.clone(),
                            });
                        }
                    }
                } else {
                    var_domains.insert(var.clone(), domain.clone());
                }

                body.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Imply(l, r)
            | TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                l.collect_and_validate_domains(registry, var_domains)?;
                r.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                left.collect_and_validate_domains(registry, var_domains)?;
                right.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                premise.collect_and_validate_domains(registry, var_domains)?;
                conclusion.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Not(e)
            | TLExpr::Score(e)
            | TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e)
            | TLExpr::Box(e)
            | TLExpr::Diamond(e)
            | TLExpr::Next(e)
            | TLExpr::Eventually(e)
            | TLExpr::Always(e) => {
                e.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::FuzzyNot { expr, .. } => {
                expr.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::WeightedRule { rule, .. } => {
                rule.collect_and_validate_domains(registry, var_domains)?;
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
                before.collect_and_validate_domains(registry, var_domains)?;
                after.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, expr) in alternatives {
                    expr.collect_and_validate_domains(registry, var_domains)?;
                }
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_and_validate_domains(registry, var_domains)?;
                then_branch.collect_and_validate_domains(registry, var_domains)?;
                else_branch.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Let { value, body, .. } => {
                value.collect_and_validate_domains(registry, var_domains)?;
                body.collect_and_validate_domains(registry, var_domains)?;
            }
            // Beta.1 enhancements
            TLExpr::Lambda { body, .. } => {
                // Lambda introduces a local binding, no domain validation
                body.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Apply { function, argument } => {
                function.collect_and_validate_domains(registry, var_domains)?;
                argument.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::SetMembership { element, set }
            | TLExpr::SetUnion {
                left: element,
                right: set,
            }
            | TLExpr::SetIntersection {
                left: element,
                right: set,
            }
            | TLExpr::SetDifference {
                left: element,
                right: set,
            } => {
                element.collect_and_validate_domains(registry, var_domains)?;
                set.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::SetCardinality { set } => {
                set.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::EmptySet => {
                // No domain validation needed
            }
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => {
                registry.validate_domain(domain)?;
                if let Some(existing_domain) = var_domains.get(var) {
                    if existing_domain != domain
                        && !registry.are_compatible(existing_domain, domain)?
                    {
                        return Err(IrError::VariableDomainMismatch {
                            var: var.clone(),
                            expected: existing_domain.clone(),
                            actual: domain.clone(),
                        });
                    }
                } else {
                    var_domains.insert(var.clone(), domain.clone());
                }
                condition.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::CountingExists {
                var, domain, body, ..
            }
            | TLExpr::CountingForAll {
                var, domain, body, ..
            }
            | TLExpr::ExactCount {
                var, domain, body, ..
            }
            | TLExpr::Majority { var, domain, body } => {
                registry.validate_domain(domain)?;
                if let Some(existing_domain) = var_domains.get(var) {
                    if existing_domain != domain
                        && !registry.are_compatible(existing_domain, domain)?
                    {
                        return Err(IrError::VariableDomainMismatch {
                            var: var.clone(),
                            expected: existing_domain.clone(),
                            actual: domain.clone(),
                        });
                    }
                } else {
                    var_domains.insert(var.clone(), domain.clone());
                }
                body.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::LeastFixpoint { body, .. } | TLExpr::GreatestFixpoint { body, .. } => {
                body.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Nominal { .. } => {
                // No domain validation needed
            }
            TLExpr::At { formula, .. } => {
                formula.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                formula.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::AllDifferent { .. } => {
                // Variables, no domain validation here
            }
            TLExpr::GlobalCardinality { values, .. } => {
                for val in values {
                    val.collect_and_validate_domains(registry, var_domains)?;
                }
            }
            TLExpr::Abducible { .. } => {
                // No domain validation needed
            }
            TLExpr::Explain { formula } => {
                formula.collect_and_validate_domains(registry, var_domains)?;
            }
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {
                // No domain validation needed for predicates and constants
            }
        }
        Ok(())
    }

    /// Extract all domains referenced in this expression.
    pub fn referenced_domains(&self) -> Vec<String> {
        let mut domains = Vec::new();
        self.collect_domains(&mut domains);
        domains.sort();
        domains.dedup();
        domains
    }

    fn collect_domains(&self, domains: &mut Vec<String>) {
        match self {
            TLExpr::Exists { domain, body, .. }
            | TLExpr::ForAll { domain, body, .. }
            | TLExpr::SoftExists { domain, body, .. }
            | TLExpr::SoftForAll { domain, body, .. }
            | TLExpr::Aggregate { domain, body, .. } => {
                domains.push(domain.clone());
                body.collect_domains(domains);
            }
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Imply(l, r)
            | TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                l.collect_domains(domains);
                r.collect_domains(domains);
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                left.collect_domains(domains);
                right.collect_domains(domains);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                premise.collect_domains(domains);
                conclusion.collect_domains(domains);
            }
            TLExpr::Not(e)
            | TLExpr::Score(e)
            | TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e)
            | TLExpr::Box(e)
            | TLExpr::Diamond(e)
            | TLExpr::Next(e)
            | TLExpr::Eventually(e)
            | TLExpr::Always(e) => {
                e.collect_domains(domains);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                expr.collect_domains(domains);
            }
            TLExpr::WeightedRule { rule, .. } => {
                rule.collect_domains(domains);
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
                before.collect_domains(domains);
                after.collect_domains(domains);
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, expr) in alternatives {
                    expr.collect_domains(domains);
                }
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_domains(domains);
                then_branch.collect_domains(domains);
                else_branch.collect_domains(domains);
            }
            TLExpr::Let { value, body, .. } => {
                value.collect_domains(domains);
                body.collect_domains(domains);
            }
            // Beta.1 enhancements
            TLExpr::Lambda { body, .. } => {
                body.collect_domains(domains);
            }
            TLExpr::Apply { function, argument } => {
                function.collect_domains(domains);
                argument.collect_domains(domains);
            }
            TLExpr::SetMembership { element, set }
            | TLExpr::SetUnion {
                left: element,
                right: set,
            }
            | TLExpr::SetIntersection {
                left: element,
                right: set,
            }
            | TLExpr::SetDifference {
                left: element,
                right: set,
            } => {
                element.collect_domains(domains);
                set.collect_domains(domains);
            }
            TLExpr::SetCardinality { set } => {
                set.collect_domains(domains);
            }
            TLExpr::EmptySet => {}
            TLExpr::SetComprehension {
                domain, condition, ..
            } => {
                domains.push(domain.clone());
                condition.collect_domains(domains);
            }
            TLExpr::CountingExists { domain, body, .. }
            | TLExpr::CountingForAll { domain, body, .. }
            | TLExpr::ExactCount { domain, body, .. }
            | TLExpr::Majority { domain, body, .. } => {
                domains.push(domain.clone());
                body.collect_domains(domains);
            }
            TLExpr::LeastFixpoint { body, .. } | TLExpr::GreatestFixpoint { body, .. } => {
                body.collect_domains(domains);
            }
            TLExpr::Nominal { .. } => {}
            TLExpr::At { formula, .. } => {
                formula.collect_domains(domains);
            }
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                formula.collect_domains(domains);
            }
            TLExpr::AllDifferent { .. } => {}
            TLExpr::GlobalCardinality { values, .. } => {
                for val in values {
                    val.collect_domains(domains);
                }
            }
            TLExpr::Abducible { .. } => {}
            TLExpr::Explain { formula } => {
                formula.collect_domains(domains);
            }
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::Term;

    #[test]
    fn test_validate_domains_success() {
        let registry = DomainRegistry::with_builtins();

        let expr = TLExpr::exists("x", "Int", TLExpr::pred("P", vec![Term::var("x")]));

        assert!(expr.validate_domains(&registry).is_ok());
    }

    #[test]
    fn test_validate_domains_not_found() {
        let registry = DomainRegistry::new();

        let expr = TLExpr::exists(
            "x",
            "UnknownDomain",
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        assert!(expr.validate_domains(&registry).is_err());
    }

    #[test]
    fn test_validate_domains_consistent_usage() {
        let registry = DomainRegistry::with_builtins();

        // ∃x:Int. ∀x:Int. P(x) - same variable, same domain
        let expr = TLExpr::exists(
            "x",
            "Int",
            TLExpr::forall("x", "Int", TLExpr::pred("P", vec![Term::var("x")])),
        );

        assert!(expr.validate_domains(&registry).is_ok());
    }

    #[test]
    fn test_validate_domains_incompatible() {
        let registry = DomainRegistry::with_builtins();

        // ∃x:Int. ∀x:Bool. P(x) - same variable, incompatible domains
        let expr = TLExpr::exists(
            "x",
            "Int",
            TLExpr::forall("x", "Bool", TLExpr::pred("P", vec![Term::var("x")])),
        );

        assert!(expr.validate_domains(&registry).is_err());
    }

    #[test]
    fn test_referenced_domains() {
        let expr = TLExpr::exists(
            "x",
            "Int",
            TLExpr::forall("y", "Real", TLExpr::pred("P", vec![Term::var("x")])),
        );

        let domains = expr.referenced_domains();
        assert_eq!(domains, vec!["Int", "Real"]);
    }

    #[test]
    fn test_referenced_domains_dedup() {
        let expr = TLExpr::and(
            TLExpr::exists("x", "Int", TLExpr::pred("P", vec![Term::var("x")])),
            TLExpr::exists("y", "Int", TLExpr::pred("Q", vec![Term::var("y")])),
        );

        let domains = expr.referenced_domains();
        assert_eq!(domains, vec!["Int"]);
    }
}
