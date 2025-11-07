//! Expression analysis (free variables, predicate collection).

use std::collections::{HashMap, HashSet};

use crate::term::Term;

use super::TLExpr;

impl TLExpr {
    /// Collect all free variables in this expression
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars, &HashSet::new());
        vars
    }

    pub(crate) fn collect_free_vars(&self, vars: &mut HashSet<String>, bound: &HashSet<String>) {
        match self {
            TLExpr::Pred { args, .. } => {
                for arg in args {
                    if let Term::Var(v) = arg {
                        if !bound.contains(v) {
                            vars.insert(v.clone());
                        }
                    }
                }
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
                l.collect_free_vars(vars, bound);
                r.collect_free_vars(vars, bound);
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                left.collect_free_vars(vars, bound);
                right.collect_free_vars(vars, bound);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                premise.collect_free_vars(vars, bound);
                conclusion.collect_free_vars(vars, bound);
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
                e.collect_free_vars(vars, bound);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                expr.collect_free_vars(vars, bound);
            }
            TLExpr::WeightedRule { rule, .. } => {
                rule.collect_free_vars(vars, bound);
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
                before.collect_free_vars(vars, bound);
                after.collect_free_vars(vars, bound);
            }
            TLExpr::Exists { var, body, .. }
            | TLExpr::ForAll { var, body, .. }
            | TLExpr::SoftExists { var, body, .. }
            | TLExpr::SoftForAll { var, body, .. } => {
                let mut new_bound = bound.clone();
                new_bound.insert(var.clone());
                body.collect_free_vars(vars, &new_bound);
            }
            TLExpr::Aggregate {
                var,
                body,
                group_by,
                ..
            } => {
                let mut new_bound = bound.clone();
                new_bound.insert(var.clone());
                body.collect_free_vars(vars, &new_bound);
                // Group-by variables are free if not already bound
                if let Some(group_vars) = group_by {
                    for gv in group_vars {
                        if !bound.contains(gv) {
                            vars.insert(gv.clone());
                        }
                    }
                }
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_free_vars(vars, bound);
                then_branch.collect_free_vars(vars, bound);
                else_branch.collect_free_vars(vars, bound);
            }
            TLExpr::Let { var, value, body } => {
                // First collect free vars from the value expression
                value.collect_free_vars(vars, bound);
                // Then collect from body with the variable bound
                let mut new_bound = bound.clone();
                new_bound.insert(var.clone());
                body.collect_free_vars(vars, &new_bound);
            }
            TLExpr::Constant(_) => {
                // No free variables in constants
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, expr) in alternatives {
                    expr.collect_free_vars(vars, bound);
                }
            }
        }
    }

    /// Collect all predicates and their arities
    pub fn all_predicates(&self) -> HashMap<String, usize> {
        let mut preds = HashMap::new();
        self.collect_predicates(&mut preds);
        preds
    }

    pub(crate) fn collect_predicates(&self, preds: &mut HashMap<String, usize>) {
        match self {
            TLExpr::Pred { name, args } => {
                preds.entry(name.clone()).or_insert(args.len());
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
                l.collect_predicates(preds);
                r.collect_predicates(preds);
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                left.collect_predicates(preds);
                right.collect_predicates(preds);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                premise.collect_predicates(preds);
                conclusion.collect_predicates(preds);
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
                e.collect_predicates(preds);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                expr.collect_predicates(preds);
            }
            TLExpr::WeightedRule { rule, .. } => {
                rule.collect_predicates(preds);
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
                before.collect_predicates(preds);
                after.collect_predicates(preds);
            }
            TLExpr::Exists { body, .. }
            | TLExpr::ForAll { body, .. }
            | TLExpr::SoftExists { body, .. }
            | TLExpr::SoftForAll { body, .. } => {
                body.collect_predicates(preds);
            }
            TLExpr::Aggregate { body, .. } => {
                body.collect_predicates(preds);
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_predicates(preds);
                then_branch.collect_predicates(preds);
                else_branch.collect_predicates(preds);
            }
            TLExpr::Let { value, body, .. } => {
                value.collect_predicates(preds);
                body.collect_predicates(preds);
            }
            TLExpr::Constant(_) => {
                // No predicates in constants
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, expr) in alternatives {
                    expr.collect_predicates(preds);
                }
            }
        }
    }
}
