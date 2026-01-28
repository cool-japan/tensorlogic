//! Expression rewriting system with pattern matching.
//!
//! This module provides a powerful rewriting system that allows users to define
//! custom transformation rules for TensorLogic expressions.

use std::collections::HashMap;

use super::TLExpr;

/// A pattern that can match against expressions.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Match any expression and bind it to a variable
    Var(String),
    /// Match a specific constant value
    Constant(f64),
    /// Match a predicate with a specific name
    Pred { name: String, args: Vec<Pattern> },
    /// Match an AND expression
    And(Box<Pattern>, Box<Pattern>),
    /// Match an OR expression
    Or(Box<Pattern>, Box<Pattern>),
    /// Match a NOT expression
    Not(Box<Pattern>),
    /// Match an implication
    Imply(Box<Pattern>, Box<Pattern>),
    /// Match any expression (wildcard)
    Any,
}

impl Pattern {
    /// Create a variable pattern.
    pub fn var(name: impl Into<String>) -> Self {
        Pattern::Var(name.into())
    }

    /// Create a constant pattern.
    pub fn constant(value: f64) -> Self {
        Pattern::Constant(value)
    }

    /// Create a wildcard pattern.
    pub fn any() -> Self {
        Pattern::Any
    }

    /// Create a predicate pattern.
    pub fn pred(name: impl Into<String>, args: Vec<Pattern>) -> Self {
        Pattern::Pred {
            name: name.into(),
            args,
        }
    }

    /// Create an AND pattern.
    pub fn and(left: Pattern, right: Pattern) -> Self {
        Pattern::And(Box::new(left), Box::new(right))
    }

    /// Create an OR pattern.
    pub fn or(left: Pattern, right: Pattern) -> Self {
        Pattern::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT pattern.
    pub fn negation(pattern: Pattern) -> Self {
        Pattern::Not(Box::new(pattern))
    }

    /// Create an implication pattern.
    pub fn imply(left: Pattern, right: Pattern) -> Self {
        Pattern::Imply(Box::new(left), Box::new(right))
    }

    /// Try to match this pattern against an expression, returning bindings if successful.
    pub fn matches(&self, expr: &TLExpr) -> Option<HashMap<String, TLExpr>> {
        let mut bindings = HashMap::new();
        if self.matches_recursive(expr, &mut bindings) {
            Some(bindings)
        } else {
            None
        }
    }

    fn matches_recursive(&self, expr: &TLExpr, bindings: &mut HashMap<String, TLExpr>) -> bool {
        match (self, expr) {
            // Wildcard matches anything
            (Pattern::Any, _) => true,

            // Variable pattern: bind if not already bound, or check if bound value matches
            (Pattern::Var(var_name), _) => {
                if let Some(bound_expr) = bindings.get(var_name) {
                    bound_expr == expr
                } else {
                    bindings.insert(var_name.clone(), expr.clone());
                    true
                }
            }

            // Constant pattern
            (Pattern::Constant(pv), TLExpr::Constant(ev)) => (pv - ev).abs() < f64::EPSILON,

            // Predicate pattern
            (
                Pattern::Pred {
                    name: pname,
                    args: pargs,
                },
                TLExpr::Pred {
                    name: ename,
                    args: eargs,
                },
            ) => {
                if pname != ename || pargs.len() != eargs.len() {
                    return false;
                }
                // Note: We're matching predicate arguments structurally here
                // For a more sophisticated system, we'd need term patterns
                pargs.len() == eargs.len()
            }

            // Binary operators
            (Pattern::And(pl, pr), TLExpr::And(el, er))
            | (Pattern::Or(pl, pr), TLExpr::Or(el, er))
            | (Pattern::Imply(pl, pr), TLExpr::Imply(el, er)) => {
                pl.matches_recursive(el, bindings) && pr.matches_recursive(er, bindings)
            }

            // Unary operators
            (Pattern::Not(p), TLExpr::Not(e)) => p.matches_recursive(e, bindings),

            _ => false,
        }
    }
}

/// A rewrite rule that transforms expressions matching a pattern into a template.
#[derive(Clone, Debug)]
pub struct RewriteRule {
    /// The pattern to match
    pub pattern: Pattern,
    /// Function to generate the replacement expression from bindings
    pub template: fn(&HashMap<String, TLExpr>) -> TLExpr,
    /// Optional name for debugging
    pub name: Option<String>,
}

impl RewriteRule {
    /// Create a new rewrite rule.
    pub fn new(pattern: Pattern, template: fn(&HashMap<String, TLExpr>) -> TLExpr) -> Self {
        Self {
            pattern,
            template,
            name: None,
        }
    }

    /// Create a named rewrite rule.
    pub fn named(
        name: impl Into<String>,
        pattern: Pattern,
        template: fn(&HashMap<String, TLExpr>) -> TLExpr,
    ) -> Self {
        Self {
            pattern,
            template,
            name: Some(name.into()),
        }
    }

    /// Try to apply this rule to an expression.
    pub fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        self.pattern
            .matches(expr)
            .map(|bindings| (self.template)(&bindings))
    }
}

/// A collection of rewrite rules that can be applied to expressions.
#[derive(Clone, Debug, Default)]
pub struct RewriteSystem {
    rules: Vec<RewriteRule>,
}

impl RewriteSystem {
    /// Create a new empty rewrite system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the system.
    pub fn add_rule(mut self, rule: RewriteRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Create a system with common logical equivalences.
    pub fn with_logic_equivalences() -> Self {
        let mut system = Self::new();

        // Double negation elimination: ¬¬A → A
        system = system.add_rule(RewriteRule::named(
            "double_negation",
            Pattern::negation(Pattern::negation(Pattern::var("A"))),
            |bindings| bindings.get("A").unwrap().clone(),
        ));

        // De Morgan's laws: ¬(A ∧ B) → ¬A ∨ ¬B
        system = system.add_rule(RewriteRule::named(
            "demorgan_and",
            Pattern::negation(Pattern::and(Pattern::var("A"), Pattern::var("B"))),
            |bindings| {
                TLExpr::or(
                    TLExpr::negate(bindings.get("A").unwrap().clone()),
                    TLExpr::negate(bindings.get("B").unwrap().clone()),
                )
            },
        ));

        // De Morgan's laws: ¬(A ∨ B) → ¬A ∧ ¬B
        system = system.add_rule(RewriteRule::named(
            "demorgan_or",
            Pattern::negation(Pattern::or(Pattern::var("A"), Pattern::var("B"))),
            |bindings| {
                TLExpr::and(
                    TLExpr::negate(bindings.get("A").unwrap().clone()),
                    TLExpr::negate(bindings.get("B").unwrap().clone()),
                )
            },
        ));

        // Implication expansion: A → B ≡ ¬A ∨ B
        system = system.add_rule(RewriteRule::named(
            "implication_expansion",
            Pattern::imply(Pattern::var("A"), Pattern::var("B")),
            |bindings| {
                TLExpr::or(
                    TLExpr::negate(bindings.get("A").unwrap().clone()),
                    bindings.get("B").unwrap().clone(),
                )
            },
        ));

        system
    }

    /// Try to apply the first matching rule to an expression.
    pub fn apply_once(&self, expr: &TLExpr) -> Option<TLExpr> {
        for rule in &self.rules {
            if let Some(result) = rule.apply(expr) {
                return Some(result);
            }
        }
        None
    }

    /// Apply rules recursively to an expression and all its subexpressions.
    pub fn apply_recursive(&self, expr: &TLExpr) -> TLExpr {
        // First, try to apply a rule at the top level
        if let Some(rewritten) = self.apply_once(expr) {
            return self.apply_recursive(&rewritten);
        }

        // If no rule applies, recurse into subexpressions
        match expr {
            TLExpr::And(l, r) => TLExpr::and(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Or(l, r) => TLExpr::or(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Not(e) => TLExpr::negate(self.apply_recursive(e)),
            TLExpr::Imply(l, r) => TLExpr::imply(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Score(e) => TLExpr::score(self.apply_recursive(e)),

            // Arithmetic
            TLExpr::Add(l, r) => TLExpr::add(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Sub(l, r) => TLExpr::sub(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Mul(l, r) => TLExpr::mul(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Div(l, r) => TLExpr::div(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Pow(l, r) => TLExpr::pow(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Mod(l, r) => TLExpr::modulo(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Min(l, r) => TLExpr::min(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Max(l, r) => TLExpr::max(self.apply_recursive(l), self.apply_recursive(r)),

            // Comparison
            TLExpr::Eq(l, r) => TLExpr::eq(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Lt(l, r) => TLExpr::lt(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Gt(l, r) => TLExpr::gt(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Lte(l, r) => TLExpr::lte(self.apply_recursive(l), self.apply_recursive(r)),
            TLExpr::Gte(l, r) => TLExpr::gte(self.apply_recursive(l), self.apply_recursive(r)),

            // Mathematical functions
            TLExpr::Abs(e) => TLExpr::abs(self.apply_recursive(e)),
            TLExpr::Floor(e) => TLExpr::floor(self.apply_recursive(e)),
            TLExpr::Ceil(e) => TLExpr::ceil(self.apply_recursive(e)),
            TLExpr::Round(e) => TLExpr::round(self.apply_recursive(e)),
            TLExpr::Sqrt(e) => TLExpr::sqrt(self.apply_recursive(e)),
            TLExpr::Exp(e) => TLExpr::exp(self.apply_recursive(e)),
            TLExpr::Log(e) => TLExpr::log(self.apply_recursive(e)),
            TLExpr::Sin(e) => TLExpr::sin(self.apply_recursive(e)),
            TLExpr::Cos(e) => TLExpr::cos(self.apply_recursive(e)),
            TLExpr::Tan(e) => TLExpr::tan(self.apply_recursive(e)),

            // Modal/Temporal
            TLExpr::Box(e) => TLExpr::modal_box(self.apply_recursive(e)),
            TLExpr::Diamond(e) => TLExpr::modal_diamond(self.apply_recursive(e)),
            TLExpr::Next(e) => TLExpr::next(self.apply_recursive(e)),
            TLExpr::Eventually(e) => TLExpr::eventually(self.apply_recursive(e)),
            TLExpr::Always(e) => TLExpr::always(self.apply_recursive(e)),
            TLExpr::Until { before, after } => {
                TLExpr::until(self.apply_recursive(before), self.apply_recursive(after))
            }
            TLExpr::Release { released, releaser } => TLExpr::release(
                self.apply_recursive(released),
                self.apply_recursive(releaser),
            ),
            TLExpr::WeakUntil { before, after } => {
                TLExpr::weak_until(self.apply_recursive(before), self.apply_recursive(after))
            }
            TLExpr::StrongRelease { released, releaser } => TLExpr::strong_release(
                self.apply_recursive(released),
                self.apply_recursive(releaser),
            ),

            // Quantifiers
            TLExpr::Exists { var, domain, body } => {
                TLExpr::exists(var.clone(), domain.clone(), self.apply_recursive(body))
            }
            TLExpr::ForAll { var, domain, body } => {
                TLExpr::forall(var.clone(), domain.clone(), self.apply_recursive(body))
            }
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => TLExpr::soft_exists(
                var.clone(),
                domain.clone(),
                self.apply_recursive(body),
                *temperature,
            ),
            TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature,
            } => TLExpr::soft_forall(
                var.clone(),
                domain.clone(),
                self.apply_recursive(body),
                *temperature,
            ),

            // Aggregation
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                if let Some(group_vars) = group_by {
                    TLExpr::aggregate_with_group_by(
                        op.clone(),
                        var.clone(),
                        domain.clone(),
                        self.apply_recursive(body),
                        group_vars.clone(),
                    )
                } else {
                    TLExpr::aggregate(
                        op.clone(),
                        var.clone(),
                        domain.clone(),
                        self.apply_recursive(body),
                    )
                }
            }

            // Control flow
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => TLExpr::if_then_else(
                self.apply_recursive(condition),
                self.apply_recursive(then_branch),
                self.apply_recursive(else_branch),
            ),
            TLExpr::Let { var, value, body } => TLExpr::let_binding(
                var.clone(),
                self.apply_recursive(value),
                self.apply_recursive(body),
            ),

            // Fuzzy logic
            TLExpr::TNorm { kind, left, right } => TLExpr::tnorm(
                *kind,
                self.apply_recursive(left),
                self.apply_recursive(right),
            ),
            TLExpr::TCoNorm { kind, left, right } => TLExpr::tconorm(
                *kind,
                self.apply_recursive(left),
                self.apply_recursive(right),
            ),
            TLExpr::FuzzyNot { kind, expr } => TLExpr::fuzzy_not(*kind, self.apply_recursive(expr)),
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => TLExpr::fuzzy_imply(
                *kind,
                self.apply_recursive(premise),
                self.apply_recursive(conclusion),
            ),

            // Probabilistic
            TLExpr::WeightedRule { weight, rule } => {
                TLExpr::weighted_rule(*weight, self.apply_recursive(rule))
            }
            TLExpr::ProbabilisticChoice { alternatives } => TLExpr::probabilistic_choice(
                alternatives
                    .iter()
                    .map(|(p, e)| (*p, self.apply_recursive(e)))
                    .collect(),
            ),

            // Beta.1 enhancements: recurse into subexpressions
            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => TLExpr::lambda(var.clone(), var_type.clone(), self.apply_recursive(body)),
            TLExpr::Apply { function, argument } => TLExpr::apply(
                self.apply_recursive(function),
                self.apply_recursive(argument),
            ),
            TLExpr::SetMembership { element, set } => {
                TLExpr::set_membership(self.apply_recursive(element), self.apply_recursive(set))
            }
            TLExpr::SetUnion { left, right } => {
                TLExpr::set_union(self.apply_recursive(left), self.apply_recursive(right))
            }
            TLExpr::SetIntersection { left, right } => {
                TLExpr::set_intersection(self.apply_recursive(left), self.apply_recursive(right))
            }
            TLExpr::SetDifference { left, right } => {
                TLExpr::set_difference(self.apply_recursive(left), self.apply_recursive(right))
            }
            TLExpr::SetCardinality { set } => TLExpr::set_cardinality(self.apply_recursive(set)),
            TLExpr::EmptySet => expr.clone(),
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => TLExpr::set_comprehension(
                var.clone(),
                domain.clone(),
                self.apply_recursive(condition),
            ),
            TLExpr::CountingExists {
                var,
                domain,
                body,
                min_count,
            } => TLExpr::counting_exists(
                var.clone(),
                domain.clone(),
                self.apply_recursive(body),
                *min_count,
            ),
            TLExpr::CountingForAll {
                var,
                domain,
                body,
                min_count,
            } => TLExpr::counting_forall(
                var.clone(),
                domain.clone(),
                self.apply_recursive(body),
                *min_count,
            ),
            TLExpr::ExactCount {
                var,
                domain,
                body,
                count,
            } => TLExpr::exact_count(
                var.clone(),
                domain.clone(),
                self.apply_recursive(body),
                *count,
            ),
            TLExpr::Majority { var, domain, body } => {
                TLExpr::majority(var.clone(), domain.clone(), self.apply_recursive(body))
            }
            TLExpr::LeastFixpoint { var, body } => {
                TLExpr::least_fixpoint(var.clone(), self.apply_recursive(body))
            }
            TLExpr::GreatestFixpoint { var, body } => {
                TLExpr::greatest_fixpoint(var.clone(), self.apply_recursive(body))
            }
            TLExpr::Nominal { .. } => expr.clone(),
            TLExpr::At { nominal, formula } => {
                TLExpr::at(nominal.clone(), self.apply_recursive(formula))
            }
            TLExpr::Somewhere { formula } => TLExpr::somewhere(self.apply_recursive(formula)),
            TLExpr::Everywhere { formula } => TLExpr::everywhere(self.apply_recursive(formula)),
            TLExpr::AllDifferent { .. } => expr.clone(),
            TLExpr::GlobalCardinality {
                variables,
                values,
                min_occurrences,
                max_occurrences,
            } => TLExpr::global_cardinality(
                variables.clone(),
                values.iter().map(|v| self.apply_recursive(v)).collect(),
                min_occurrences.clone(),
                max_occurrences.clone(),
            ),
            TLExpr::Abducible { .. } => expr.clone(),
            TLExpr::Explain { formula } => TLExpr::explain(self.apply_recursive(formula)),

            // Leaves - no recursion needed
            TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
        }
    }

    /// Apply rules until no more changes occur (fixed point).
    pub fn apply_until_fixpoint(&self, expr: &TLExpr) -> TLExpr {
        let mut current = expr.clone();
        loop {
            let next = self.apply_recursive(&current);
            if next == current {
                return current;
            }
            current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_pattern_var_match() {
        let pattern = Pattern::var("x");
        let expr = TLExpr::pred("P", vec![Term::var("a")]);

        let bindings = pattern.matches(&expr).unwrap();
        assert_eq!(bindings.get("x"), Some(&expr));
    }

    #[test]
    fn test_pattern_constant_match() {
        let pattern = Pattern::constant(42.0);
        let expr = TLExpr::constant(42.0);

        assert!(pattern.matches(&expr).is_some());
    }

    #[test]
    fn test_pattern_and_match() {
        let pattern = Pattern::and(Pattern::var("A"), Pattern::var("B"));
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y")]),
        );

        let bindings = pattern.matches(&expr).unwrap();
        assert!(bindings.contains_key("A"));
        assert!(bindings.contains_key("B"));
    }

    #[test]
    fn test_pattern_not_match() {
        let pattern = Pattern::negation(Pattern::var("A"));
        let expr = TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")]));

        let bindings = pattern.matches(&expr).unwrap();
        assert!(bindings.contains_key("A"));
    }

    #[test]
    fn test_double_negation_rule() {
        let rule = RewriteRule::new(
            Pattern::negation(Pattern::negation(Pattern::var("A"))),
            |bindings| bindings.get("A").unwrap().clone(),
        );

        let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));
        let result = rule.apply(&expr).unwrap();

        assert!(matches!(result, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_rewrite_system_double_negation() {
        let system = RewriteSystem::new().add_rule(RewriteRule::new(
            Pattern::negation(Pattern::negation(Pattern::var("A"))),
            |bindings| bindings.get("A").unwrap().clone(),
        ));

        let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));
        let result = system.apply_recursive(&expr);

        assert!(matches!(result, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_logic_equivalences_system() {
        let system = RewriteSystem::with_logic_equivalences();

        // Test double negation
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));
        let result = system.apply_recursive(&expr);
        assert!(matches!(result, TLExpr::Pred { .. }));

        // Test De Morgan's law: ¬(A ∧ B) → ¬A ∨ ¬B
        let expr = TLExpr::negate(TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y")]),
        ));
        let result = system.apply_recursive(&expr);
        assert!(matches!(result, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_nested_rewriting() {
        let system = RewriteSystem::with_logic_equivalences();

        // ¬(¬¬P ∧ Q) should be rewritten to ¬(P ∧ Q) then to ¬P ∨ ¬Q
        let expr = TLExpr::negate(TLExpr::and(
            TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")]))),
            TLExpr::pred("Q", vec![Term::var("y")]),
        ));

        let result = system.apply_until_fixpoint(&expr);
        // Should be ¬P ∨ ¬Q after full rewriting
        assert!(matches!(result, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_implication_expansion() {
        let system = RewriteSystem::with_logic_equivalences();

        // P → Q should expand to ¬P ∨ Q
        let expr = TLExpr::imply(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y")]),
        );

        let result = system.apply_recursive(&expr);
        assert!(matches!(result, TLExpr::Or(_, _)));
    }
}
