//! Expression validation (arity checking).

use std::collections::HashMap;

use super::TLExpr;

impl TLExpr {
    /// Validate that all predicates with the same name have consistent arity
    pub fn validate_arity(&self) -> Result<(), String> {
        self.validate_arity_recursive(&HashMap::new())
    }

    fn validate_arity_recursive(&self, seen: &HashMap<String, usize>) -> Result<(), String> {
        match self {
            TLExpr::Pred { name, args } => {
                if let Some(&expected_arity) = seen.get(name) {
                    if expected_arity != args.len() {
                        return Err(format!(
                            "Predicate '{}' has inconsistent arity: expected {}, found {}",
                            name,
                            expected_arity,
                            args.len()
                        ));
                    }
                }
                let mut new_seen = seen.clone();
                new_seen.insert(name.clone(), args.len());
                Ok(())
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
                let mut new_seen = seen.clone();

                l.collect_and_check_arity(&mut new_seen)?;
                r.collect_and_check_arity(&mut new_seen)?;

                Ok(())
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
            | TLExpr::Always(e) => e.validate_arity_recursive(seen),
            TLExpr::Until { before, after } => {
                let mut new_seen = seen.clone();
                before.collect_and_check_arity(&mut new_seen)?;
                after.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                body.validate_arity_recursive(seen)
            }
            TLExpr::Aggregate { body, .. } => body.validate_arity_recursive(seen),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut new_seen = seen.clone();
                condition.collect_and_check_arity(&mut new_seen)?;
                then_branch.collect_and_check_arity(&mut new_seen)?;
                else_branch.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }
            TLExpr::Let { value, body, .. } => {
                let mut new_seen = seen.clone();
                value.collect_and_check_arity(&mut new_seen)?;
                body.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }

            // Fuzzy logic operators
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                let mut new_seen = seen.clone();
                left.collect_and_check_arity(&mut new_seen)?;
                right.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }
            TLExpr::FuzzyNot { expr, .. } => expr.validate_arity_recursive(seen),
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                let mut new_seen = seen.clone();
                premise.collect_and_check_arity(&mut new_seen)?;
                conclusion.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }

            // Probabilistic operators
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                body.validate_arity_recursive(seen)
            }
            TLExpr::WeightedRule { rule, .. } => rule.validate_arity_recursive(seen),
            TLExpr::ProbabilisticChoice { alternatives } => {
                let mut new_seen = seen.clone();
                for (_, expr) in alternatives {
                    expr.collect_and_check_arity(&mut new_seen)?;
                }
                Ok(())
            }

            // Extended temporal logic
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                let mut new_seen = seen.clone();
                released.collect_and_check_arity(&mut new_seen)?;
                releaser.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }

            // Alpha.3 enhancements
            TLExpr::Lambda { body, .. } => body.validate_arity_recursive(seen),
            TLExpr::Apply { function, argument } => {
                let mut new_seen = seen.clone();
                function.collect_and_check_arity(&mut new_seen)?;
                argument.collect_and_check_arity(&mut new_seen)?;
                Ok(())
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
                let mut new_seen = seen.clone();
                element.collect_and_check_arity(&mut new_seen)?;
                set.collect_and_check_arity(&mut new_seen)?;
                Ok(())
            }
            TLExpr::SetCardinality { set } => set.validate_arity_recursive(seen),
            TLExpr::EmptySet => Ok(()),
            TLExpr::SetComprehension { condition, .. } => condition.validate_arity_recursive(seen),
            TLExpr::CountingExists { body, .. }
            | TLExpr::CountingForAll { body, .. }
            | TLExpr::ExactCount { body, .. }
            | TLExpr::Majority { body, .. } => body.validate_arity_recursive(seen),
            TLExpr::LeastFixpoint { body, .. } | TLExpr::GreatestFixpoint { body, .. } => {
                body.validate_arity_recursive(seen)
            }
            TLExpr::Nominal { .. } => Ok(()),
            TLExpr::At { formula, .. } => formula.validate_arity_recursive(seen),
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                formula.validate_arity_recursive(seen)
            }
            TLExpr::AllDifferent { .. } => Ok(()),
            TLExpr::GlobalCardinality { values, .. } => {
                let mut new_seen = seen.clone();
                for val in values {
                    val.collect_and_check_arity(&mut new_seen)?;
                }
                Ok(())
            }
            TLExpr::Abducible { .. } => Ok(()),
            TLExpr::Explain { formula } => formula.validate_arity_recursive(seen),

            TLExpr::Constant(_) => Ok(()),
        }
    }

    pub(crate) fn collect_and_check_arity(
        &self,
        seen: &mut HashMap<String, usize>,
    ) -> Result<(), String> {
        match self {
            TLExpr::Pred { name, args } => {
                if let Some(&expected_arity) = seen.get(name) {
                    if expected_arity != args.len() {
                        return Err(format!(
                            "Predicate '{}' has inconsistent arity: expected {}, found {}",
                            name,
                            expected_arity,
                            args.len()
                        ));
                    }
                } else {
                    seen.insert(name.clone(), args.len());
                }
                Ok(())
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
                l.collect_and_check_arity(seen)?;
                r.collect_and_check_arity(seen)?;
                Ok(())
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
            | TLExpr::Always(e) => e.collect_and_check_arity(seen),
            TLExpr::Until { before, after } => {
                before.collect_and_check_arity(seen)?;
                after.collect_and_check_arity(seen)?;
                Ok(())
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                body.collect_and_check_arity(seen)
            }
            TLExpr::Aggregate { body, .. } => body.collect_and_check_arity(seen),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_and_check_arity(seen)?;
                then_branch.collect_and_check_arity(seen)?;
                else_branch.collect_and_check_arity(seen)?;
                Ok(())
            }
            TLExpr::Let { value, body, .. } => {
                value.collect_and_check_arity(seen)?;
                body.collect_and_check_arity(seen)?;
                Ok(())
            }

            // Fuzzy logic operators
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                left.collect_and_check_arity(seen)?;
                right.collect_and_check_arity(seen)?;
                Ok(())
            }
            TLExpr::FuzzyNot { expr, .. } => expr.collect_and_check_arity(seen),
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                premise.collect_and_check_arity(seen)?;
                conclusion.collect_and_check_arity(seen)?;
                Ok(())
            }

            // Probabilistic operators
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                body.collect_and_check_arity(seen)
            }
            TLExpr::WeightedRule { rule, .. } => rule.collect_and_check_arity(seen),
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, expr) in alternatives {
                    expr.collect_and_check_arity(seen)?;
                }
                Ok(())
            }

            // Extended temporal logic
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                released.collect_and_check_arity(seen)?;
                releaser.collect_and_check_arity(seen)?;
                Ok(())
            }

            // Alpha.3 enhancements
            TLExpr::Lambda { body, .. } => body.collect_and_check_arity(seen),
            TLExpr::Apply { function, argument } => {
                function.collect_and_check_arity(seen)?;
                argument.collect_and_check_arity(seen)?;
                Ok(())
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
                element.collect_and_check_arity(seen)?;
                set.collect_and_check_arity(seen)?;
                Ok(())
            }
            TLExpr::SetCardinality { set } => set.collect_and_check_arity(seen),
            TLExpr::EmptySet => Ok(()),
            TLExpr::SetComprehension { condition, .. } => condition.collect_and_check_arity(seen),
            TLExpr::CountingExists { body, .. }
            | TLExpr::CountingForAll { body, .. }
            | TLExpr::ExactCount { body, .. }
            | TLExpr::Majority { body, .. } => body.collect_and_check_arity(seen),
            TLExpr::LeastFixpoint { body, .. } | TLExpr::GreatestFixpoint { body, .. } => {
                body.collect_and_check_arity(seen)
            }
            TLExpr::Nominal { .. } => Ok(()),
            TLExpr::At { formula, .. } => formula.collect_and_check_arity(seen),
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                formula.collect_and_check_arity(seen)
            }
            TLExpr::AllDifferent { .. } => Ok(()),
            TLExpr::GlobalCardinality { values, .. } => {
                for val in values {
                    val.collect_and_check_arity(seen)?;
                }
                Ok(())
            }
            TLExpr::Abducible { .. } => Ok(()),
            TLExpr::Explain { formula } => formula.collect_and_check_arity(seen),

            TLExpr::Constant(_) => Ok(()),
        }
    }
}
