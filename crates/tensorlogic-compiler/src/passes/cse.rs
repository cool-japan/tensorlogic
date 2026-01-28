//! Common Subexpression Elimination (CSE) for TLExpr.

use std::collections::HashMap;

use tensorlogic_ir::TLExpr;

/// CSE result containing optimized expression and statistics
#[derive(Debug, Clone)]
pub struct CseResult {
    pub optimized_expr: TLExpr,
    pub eliminated_count: usize,
}

/// Perform common subexpression elimination on a TLExpr
pub fn eliminate_common_subexpressions(expr: &TLExpr) -> CseResult {
    let mut cache: HashMap<String, TLExpr> = HashMap::new();
    let mut eliminated_count = 0;

    let optimized = cse_recursive(expr, &mut cache, &mut eliminated_count);

    CseResult {
        optimized_expr: optimized,
        eliminated_count,
    }
}

fn cse_recursive(
    expr: &TLExpr,
    cache: &mut HashMap<String, TLExpr>,
    eliminated_count: &mut usize,
) -> TLExpr {
    // Compute a hash/key for this expression
    let key = expr_to_key(expr);

    // Check if we've seen this exact subexpression before
    if let Some(cached) = cache.get(&key) {
        *eliminated_count += 1;
        return cached.clone();
    }

    // Recursively process subexpressions
    let result = match expr {
        TLExpr::Pred { .. } => {
            // Predicates are atomic, just cache them
            expr.clone()
        }
        TLExpr::And(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::and(left_opt, right_opt)
        }
        TLExpr::Or(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::or(left_opt, right_opt)
        }
        TLExpr::Imply(premise, conclusion) => {
            let premise_opt = cse_recursive(premise, cache, eliminated_count);
            let conclusion_opt = cse_recursive(conclusion, cache, eliminated_count);
            TLExpr::imply(premise_opt, conclusion_opt)
        }
        TLExpr::Not(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::negate(inner_opt)
        }
        TLExpr::Exists { var, domain, body } => {
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::exists(var, domain, body_opt)
        }
        TLExpr::ForAll { var, domain, body } => {
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::forall(var, domain, body_opt)
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::aggregate_with_group_by(
                op.clone(),
                var,
                domain,
                body_opt,
                group_by.clone().unwrap_or_default(),
            )
        }
        TLExpr::Score(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::score(inner_opt)
        }
        // Arithmetic operations
        TLExpr::Add(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::add(left_opt, right_opt)
        }
        TLExpr::Sub(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::sub(left_opt, right_opt)
        }
        TLExpr::Mul(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::mul(left_opt, right_opt)
        }
        TLExpr::Div(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::div(left_opt, right_opt)
        }
        // Comparison operations
        TLExpr::Eq(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::eq(left_opt, right_opt)
        }
        TLExpr::Lt(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::lt(left_opt, right_opt)
        }
        TLExpr::Gt(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::gt(left_opt, right_opt)
        }
        TLExpr::Lte(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::lte(left_opt, right_opt)
        }
        TLExpr::Gte(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::gte(left_opt, right_opt)
        }
        TLExpr::Pow(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::pow(left_opt, right_opt)
        }
        TLExpr::Mod(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::modulo(left_opt, right_opt)
        }
        TLExpr::Min(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::min(left_opt, right_opt)
        }
        TLExpr::Max(left, right) => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::max(left_opt, right_opt)
        }
        // Unary math operations
        TLExpr::Abs(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::abs(inner_opt)
        }
        TLExpr::Floor(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::floor(inner_opt)
        }
        TLExpr::Ceil(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::ceil(inner_opt)
        }
        TLExpr::Round(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::round(inner_opt)
        }
        TLExpr::Sqrt(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::sqrt(inner_opt)
        }
        TLExpr::Exp(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::exp(inner_opt)
        }
        TLExpr::Log(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::log(inner_opt)
        }
        TLExpr::Sin(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::sin(inner_opt)
        }
        TLExpr::Cos(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::cos(inner_opt)
        }
        TLExpr::Tan(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::tan(inner_opt)
        }
        // Let binding
        TLExpr::Let { var, value, body } => {
            let value_opt = cse_recursive(value, cache, eliminated_count);
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::let_binding(var, value_opt, body_opt)
        }
        // Conditional
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_opt = cse_recursive(condition, cache, eliminated_count);
            let then_opt = cse_recursive(then_branch, cache, eliminated_count);
            let else_opt = cse_recursive(else_branch, cache, eliminated_count);
            TLExpr::if_then_else(cond_opt, then_opt, else_opt)
        }
        // Constant
        TLExpr::Constant(_) => {
            // Constants are atomic, just cache them
            expr.clone()
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::Box(Box::new(inner_opt))
        }
        TLExpr::Diamond(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::Diamond(Box::new(inner_opt))
        }
        TLExpr::Next(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::Next(Box::new(inner_opt))
        }
        TLExpr::Eventually(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::Eventually(Box::new(inner_opt))
        }
        TLExpr::Always(inner) => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::Always(Box::new(inner_opt))
        }
        TLExpr::Until { before, after } => {
            let before_opt = cse_recursive(before, cache, eliminated_count);
            let after_opt = cse_recursive(after, cache, eliminated_count);
            TLExpr::Until {
                before: Box::new(before_opt),
                after: Box::new(after_opt),
            }
        }
        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::TNorm {
                kind: *kind,
                left: Box::new(left_opt),
                right: Box::new(right_opt),
            }
        }
        TLExpr::TCoNorm { kind, left, right } => {
            let left_opt = cse_recursive(left, cache, eliminated_count);
            let right_opt = cse_recursive(right, cache, eliminated_count);
            TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(left_opt),
                right: Box::new(right_opt),
            }
        }
        TLExpr::FuzzyNot { kind, expr: inner } => {
            let inner_opt = cse_recursive(inner, cache, eliminated_count);
            TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(inner_opt),
            }
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            let premise_opt = cse_recursive(premise, cache, eliminated_count);
            let conclusion_opt = cse_recursive(conclusion, cache, eliminated_count);
            TLExpr::FuzzyImplication {
                kind: *kind,
                premise: Box::new(premise_opt),
                conclusion: Box::new(conclusion_opt),
            }
        }
        // Soft quantifiers
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::SoftExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
                temperature: *temperature,
            }
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_opt = cse_recursive(body, cache, eliminated_count);
            TLExpr::SoftForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
                temperature: *temperature,
            }
        }
        // Weighted/probabilistic operators
        TLExpr::WeightedRule { weight, rule } => {
            let rule_opt = cse_recursive(rule, cache, eliminated_count);
            TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(rule_opt),
            }
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let alts_opt: Vec<(f64, TLExpr)> = alternatives
                .iter()
                .map(|(prob, expr)| (*prob, cse_recursive(expr, cache, eliminated_count)))
                .collect();
            TLExpr::ProbabilisticChoice {
                alternatives: alts_opt,
            }
        }
        // Extended temporal operators
        TLExpr::Release { released, releaser } => {
            let released_opt = cse_recursive(released, cache, eliminated_count);
            let releaser_opt = cse_recursive(releaser, cache, eliminated_count);
            TLExpr::Release {
                released: Box::new(released_opt),
                releaser: Box::new(releaser_opt),
            }
        }
        TLExpr::WeakUntil { before, after } => {
            let before_opt = cse_recursive(before, cache, eliminated_count);
            let after_opt = cse_recursive(after, cache, eliminated_count);
            TLExpr::WeakUntil {
                before: Box::new(before_opt),
                after: Box::new(after_opt),
            }
        }
        TLExpr::StrongRelease { released, releaser } => {
            let released_opt = cse_recursive(released, cache, eliminated_count);
            let releaser_opt = cse_recursive(releaser, cache, eliminated_count);
            TLExpr::StrongRelease {
                released: Box::new(released_opt),
                releaser: Box::new(releaser_opt),
            }
        }
        // All other expression types (enhancements)
        _ => expr.clone(),
    };

    // Cache this result
    cache.insert(key, result.clone());
    result
}

/// Convert an expression to a hashable key
fn expr_to_key(expr: &TLExpr) -> String {
    // Use debug format as a simple hash
    // In production, you'd want a proper hash function
    format!("{:?}", expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_cse_no_duplicates() {
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("y")]),
        );

        let result = eliminate_common_subexpressions(&expr);
        assert_eq!(result.eliminated_count, 0);
    }

    #[test]
    fn test_cse_duplicate_predicates() {
        // p(x) ∧ p(x) - should detect duplicate
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::and(p_x.clone(), p_x);

        let result = eliminate_common_subexpressions(&expr);
        // Should eliminate at least one duplicate
        assert!(result.eliminated_count > 0);
    }

    #[test]
    fn test_cse_nested_duplicates() {
        // (p(x) ∧ q(y)) ∧ (p(x) ∧ q(y)) - duplicate AND subexpressions
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let q_y = TLExpr::pred("q", vec![Term::var("y")]);
        let sub = TLExpr::and(p_x, q_y);
        let expr = TLExpr::and(sub.clone(), sub);

        let result = eliminate_common_subexpressions(&expr);
        assert!(result.eliminated_count > 0);
    }

    #[test]
    fn test_cse_with_quantifiers() {
        // ∃x. p(x) ∧ ∃x. p(x) - duplicate existentials
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let exists = TLExpr::exists("x", "Domain", p_x);
        let expr = TLExpr::and(exists.clone(), exists);

        let result = eliminate_common_subexpressions(&expr);
        assert!(result.eliminated_count > 0);
    }

    #[test]
    fn test_cse_preserves_semantics() {
        // Verify that CSE doesn't change the structure inappropriately
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let q_y = TLExpr::pred("q", vec![Term::var("y")]);
        let expr = TLExpr::and(p_x.clone(), q_y.clone());

        let result = eliminate_common_subexpressions(&expr);

        // Should still be an AND of two predicates
        match result.optimized_expr {
            TLExpr::And(left, right) => {
                assert!(matches!(*left, TLExpr::Pred { .. }));
                assert!(matches!(*right, TLExpr::Pred { .. }));
            }
            _ => panic!("Expected And expression"),
        }
    }

    #[test]
    fn test_cse_complex_expression() {
        // p(x) ∧ (q(y) ∨ p(x)) - p(x) appears twice
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let q_y = TLExpr::pred("q", vec![Term::var("y")]);
        let or_expr = TLExpr::or(q_y, p_x.clone());
        let expr = TLExpr::and(p_x, or_expr);

        let result = eliminate_common_subexpressions(&expr);
        assert!(result.eliminated_count > 0);
    }
}
