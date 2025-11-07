//! Expression-level optimizations (constant folding, algebraic simplification).

use super::TLExpr;

/// Constant folding: evaluate constant expressions at compile time
pub fn constant_fold(expr: &TLExpr) -> TLExpr {
    match expr {
        // Binary arithmetic operations on constants
        TLExpr::Add(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv + rv);
            }
            TLExpr::Add(Box::new(left), Box::new(right))
        }
        TLExpr::Sub(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv - rv);
            }
            TLExpr::Sub(Box::new(left), Box::new(right))
        }
        TLExpr::Mul(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv * rv);
            }
            TLExpr::Mul(Box::new(left), Box::new(right))
        }
        TLExpr::Div(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                if *rv != 0.0 {
                    return TLExpr::Constant(lv / rv);
                }
            }
            TLExpr::Div(Box::new(left), Box::new(right))
        }
        TLExpr::Pow(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.powf(*rv));
            }
            TLExpr::Pow(Box::new(left), Box::new(right))
        }
        TLExpr::Mod(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv % rv);
            }
            TLExpr::Mod(Box::new(left), Box::new(right))
        }
        TLExpr::Min(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.min(*rv));
            }
            TLExpr::Min(Box::new(left), Box::new(right))
        }
        TLExpr::Max(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.max(*rv));
            }
            TLExpr::Max(Box::new(left), Box::new(right))
        }

        // Unary mathematical operations on constants
        TLExpr::Abs(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.abs());
            }
            TLExpr::Abs(Box::new(inner))
        }
        TLExpr::Floor(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.floor());
            }
            TLExpr::Floor(Box::new(inner))
        }
        TLExpr::Ceil(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.ceil());
            }
            TLExpr::Ceil(Box::new(inner))
        }
        TLExpr::Round(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.round());
            }
            TLExpr::Round(Box::new(inner))
        }
        TLExpr::Sqrt(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                if *v >= 0.0 {
                    return TLExpr::Constant(v.sqrt());
                }
            }
            TLExpr::Sqrt(Box::new(inner))
        }
        TLExpr::Exp(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.exp());
            }
            TLExpr::Exp(Box::new(inner))
        }
        TLExpr::Log(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                if *v > 0.0 {
                    return TLExpr::Constant(v.ln());
                }
            }
            TLExpr::Log(Box::new(inner))
        }
        TLExpr::Sin(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.sin());
            }
            TLExpr::Sin(Box::new(inner))
        }
        TLExpr::Cos(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.cos());
            }
            TLExpr::Cos(Box::new(inner))
        }
        TLExpr::Tan(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.tan());
            }
            TLExpr::Tan(Box::new(inner))
        }

        TLExpr::Box(e) => TLExpr::Box(Box::new(constant_fold(e))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(constant_fold(e))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(constant_fold(e))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(constant_fold(e))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(constant_fold(e))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(constant_fold(before)),
            after: Box::new(constant_fold(after)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(constant_fold(left)),
            right: Box::new(constant_fold(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(constant_fold(left)),
            right: Box::new(constant_fold(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(constant_fold(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(constant_fold(premise)),
            conclusion: Box::new(constant_fold(conclusion)),
        },

        // Probabilistic operators
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(constant_fold(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, constant_fold(e)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(constant_fold(released)),
            releaser: Box::new(constant_fold(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(constant_fold(before)),
            after: Box::new(constant_fold(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(constant_fold(released)),
            releaser: Box::new(constant_fold(releaser)),
        },

        // Comparison operations on constants
        TLExpr::Eq(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if (lv - rv).abs() < f64::EPSILON {
                    1.0
                } else {
                    0.0
                });
            }
            TLExpr::Eq(Box::new(left), Box::new(right))
        }
        TLExpr::Lt(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv < rv { 1.0 } else { 0.0 });
            }
            TLExpr::Lt(Box::new(left), Box::new(right))
        }
        TLExpr::Gt(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv > rv { 1.0 } else { 0.0 });
            }
            TLExpr::Gt(Box::new(left), Box::new(right))
        }
        TLExpr::Lte(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv <= rv { 1.0 } else { 0.0 });
            }
            TLExpr::Lte(Box::new(left), Box::new(right))
        }
        TLExpr::Gte(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv >= rv { 1.0 } else { 0.0 });
            }
            TLExpr::Gte(Box::new(left), Box::new(right))
        }

        // Logical operations
        TLExpr::Not(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(1.0 - v);
            }
            TLExpr::Not(Box::new(inner))
        }
        TLExpr::And(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            TLExpr::And(Box::new(left), Box::new(right))
        }
        TLExpr::Or(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            TLExpr::Or(Box::new(left), Box::new(right))
        }
        TLExpr::Imply(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            TLExpr::Imply(Box::new(left), Box::new(right))
        }

        // Recursive folding for other operations
        TLExpr::Score(e) => TLExpr::Score(Box::new(constant_fold(e))),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
        },
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => TLExpr::Aggregate {
            op: op.clone(),
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
            group_by: group_by.clone(),
        },
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(constant_fold(condition)),
            then_branch: Box::new(constant_fold(then_branch)),
            else_branch: Box::new(constant_fold(else_branch)),
        },
        TLExpr::Let { var, value, body } => TLExpr::Let {
            var: var.clone(),
            value: Box::new(constant_fold(value)),
            body: Box::new(constant_fold(body)),
        },

        // Leaves - no folding needed
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

/// Algebraic simplification: apply algebraic identities and simplification rules
pub fn algebraic_simplify(expr: &TLExpr) -> TLExpr {
    match expr {
        // Addition identities
        TLExpr::Add(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x + 0 = x
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            // 0 + x = x
            if let TLExpr::Constant(0.0) = left {
                return right;
            }

            TLExpr::Add(Box::new(left), Box::new(right))
        }

        // Subtraction identities
        TLExpr::Sub(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x - 0 = x
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            // x - x = 0 (simplified form comparison)
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Sub(Box::new(left), Box::new(right))
        }

        // Multiplication identities
        TLExpr::Mul(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x * 0 = 0
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(0.0);
            }
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(0.0);
            }

            // x * 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            // 1 * x = x
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            TLExpr::Mul(Box::new(left), Box::new(right))
        }

        // Division identities
        TLExpr::Div(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x / 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }

            // 0 / x = 0 (assuming x != 0)
            if let TLExpr::Constant(0.0) = left {
                if let TLExpr::Constant(rv) = right {
                    if rv != 0.0 {
                        return TLExpr::Constant(0.0);
                    }
                }
            }

            // x / x = 1 (assuming x != 0)
            // Only apply for constants to avoid division by zero issues
            if left == right {
                if let TLExpr::Constant(v) = left {
                    if v != 0.0 {
                        return TLExpr::Constant(1.0);
                    }
                }
            }

            TLExpr::Div(Box::new(left), Box::new(right))
        }

        // Power identities
        TLExpr::Pow(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x ^ 0 = 1
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(1.0);
            }
            // x ^ 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            // 0 ^ x = 0 (for x > 0)
            if let TLExpr::Constant(0.0) = left {
                if let TLExpr::Constant(rv) = right {
                    if rv > 0.0 {
                        return TLExpr::Constant(0.0);
                    }
                }
            }
            // 1 ^ x = 1
            if let TLExpr::Constant(1.0) = left {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Pow(Box::new(left), Box::new(right))
        }

        // Double negation: NOT(NOT(x)) = x
        TLExpr::Not(e) => {
            let inner = algebraic_simplify(e);
            if let TLExpr::Not(inner_inner) = &inner {
                return *inner_inner.clone();
            }
            TLExpr::Not(Box::new(inner))
        }

        // Recursively simplify other operations
        TLExpr::Mod(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Mod(Box::new(left), Box::new(right))
        }
        TLExpr::Min(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Min(Box::new(left), Box::new(right))
        }
        TLExpr::Max(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Max(Box::new(left), Box::new(right))
        }
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(algebraic_simplify(e))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(algebraic_simplify(e))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(algebraic_simplify(e))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(algebraic_simplify(e))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(algebraic_simplify(e))),
        // Modal logic simplifications
        TLExpr::Box(e) => {
            let inner = algebraic_simplify(e);

            // □(TRUE) = TRUE, □(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Box(Box::new(inner))
        }
        TLExpr::Diamond(e) => {
            let inner = algebraic_simplify(e);

            // ◇(TRUE) = TRUE, ◇(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Diamond(Box::new(inner))
        }

        // Temporal logic simplifications
        TLExpr::Next(e) => {
            let inner = algebraic_simplify(e);

            // X(TRUE) = TRUE, X(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Next(Box::new(inner))
        }
        TLExpr::Eventually(e) => {
            let inner = algebraic_simplify(e);

            // F(TRUE) = TRUE, F(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            // Idempotence: F(F(P)) = F(P)
            if let TLExpr::Eventually(inner_inner) = &inner {
                return TLExpr::Eventually(inner_inner.clone());
            }

            TLExpr::Eventually(Box::new(inner))
        }
        TLExpr::Always(e) => {
            let inner = algebraic_simplify(e);

            // G(TRUE) = TRUE, G(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            // Idempotence: G(G(P)) = G(P)
            if let TLExpr::Always(inner_inner) = &inner {
                return TLExpr::Always(inner_inner.clone());
            }

            TLExpr::Always(Box::new(inner))
        }
        TLExpr::Until { before, after } => {
            let before_simplified = algebraic_simplify(before);
            let after_simplified = algebraic_simplify(after);

            // P U TRUE = TRUE (after becomes immediately true)
            if let TLExpr::Constant(1.0) = after_simplified {
                return TLExpr::Constant(1.0);
            }

            // FALSE U P = F(P) (before is never true, so we just wait for after)
            if let TLExpr::Constant(0.0) = before_simplified {
                return TLExpr::Eventually(Box::new(after_simplified));
            }

            TLExpr::Until {
                before: Box::new(before_simplified),
                after: Box::new(after_simplified),
            }
        }

        // Fuzzy logic operators - pass through with recursive simplification
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(algebraic_simplify(left)),
            right: Box::new(algebraic_simplify(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(algebraic_simplify(left)),
            right: Box::new(algebraic_simplify(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(algebraic_simplify(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(algebraic_simplify(premise)),
            conclusion: Box::new(algebraic_simplify(conclusion)),
        },

        // Probabilistic operators - pass through
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(algebraic_simplify(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, algebraic_simplify(e)))
                .collect(),
        },

        // Extended temporal logic - pass through
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(algebraic_simplify(released)),
            releaser: Box::new(algebraic_simplify(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(algebraic_simplify(before)),
            after: Box::new(algebraic_simplify(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(algebraic_simplify(released)),
            releaser: Box::new(algebraic_simplify(releaser)),
        },

        TLExpr::Exp(e) => TLExpr::Exp(Box::new(algebraic_simplify(e))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(algebraic_simplify(e))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(algebraic_simplify(e))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(algebraic_simplify(e))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(algebraic_simplify(e))),
        // EQ simplifications
        TLExpr::Eq(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x = x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Eq(Box::new(left), Box::new(right))
        }

        // LT simplifications
        TLExpr::Lt(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x < x → FALSE
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Lt(Box::new(left), Box::new(right))
        }

        // GT simplifications
        TLExpr::Gt(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x > x → FALSE
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Gt(Box::new(left), Box::new(right))
        }

        // LTE simplifications
        TLExpr::Lte(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x <= x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Lte(Box::new(left), Box::new(right))
        }

        // GTE simplifications
        TLExpr::Gte(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x >= x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Gte(Box::new(left), Box::new(right))
        }
        // AND logical laws
        TLExpr::And(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // Idempotence: A ∧ A = A
            if left == right {
                return left;
            }

            // Identity: A ∧ TRUE = A, TRUE ∧ A = A
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            // Annihilation: A ∧ FALSE = FALSE, FALSE ∧ A = FALSE
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(0.0);
            }
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(0.0);
            }

            // Complement: A ∧ ¬A = FALSE
            if let TLExpr::Not(inner) = &right {
                if **inner == left {
                    return TLExpr::Constant(0.0);
                }
            }
            if let TLExpr::Not(inner) = &left {
                if **inner == right {
                    return TLExpr::Constant(0.0);
                }
            }

            // Absorption: A ∧ (A ∨ B) = A
            if let TLExpr::Or(or_left, _or_right) = &right {
                if **or_left == left {
                    return left;
                }
            }
            if let TLExpr::Or(or_left, _or_right) = &left {
                if **or_left == right {
                    return right;
                }
            }

            TLExpr::And(Box::new(left), Box::new(right))
        }

        // OR logical laws
        TLExpr::Or(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // Idempotence: A ∨ A = A
            if left == right {
                return left;
            }

            // Annihilation: A ∨ TRUE = TRUE, TRUE ∨ A = TRUE
            if let TLExpr::Constant(1.0) = right {
                return TLExpr::Constant(1.0);
            }
            if let TLExpr::Constant(1.0) = left {
                return TLExpr::Constant(1.0);
            }

            // Identity: A ∨ FALSE = A, FALSE ∨ A = A
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            if let TLExpr::Constant(0.0) = left {
                return right;
            }

            // Complement: A ∨ ¬A = TRUE
            if let TLExpr::Not(inner) = &right {
                if **inner == left {
                    return TLExpr::Constant(1.0);
                }
            }
            if let TLExpr::Not(inner) = &left {
                if **inner == right {
                    return TLExpr::Constant(1.0);
                }
            }

            // Absorption: A ∨ (A ∧ B) = A
            if let TLExpr::And(and_left, _and_right) = &right {
                if **and_left == left {
                    return left;
                }
            }
            if let TLExpr::And(and_left, _and_right) = &left {
                if **and_left == right {
                    return right;
                }
            }

            TLExpr::Or(Box::new(left), Box::new(right))
        }

        // IMPLY simplifications
        TLExpr::Imply(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // TRUE → P = P
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            // FALSE → P = TRUE
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(1.0);
            }

            // P → TRUE = TRUE
            if let TLExpr::Constant(1.0) = right {
                return TLExpr::Constant(1.0);
            }

            // P → FALSE = ¬P
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::negate(left);
            }

            // P → P = TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Imply(Box::new(left), Box::new(right))
        }
        TLExpr::Score(e) => TLExpr::Score(Box::new(algebraic_simplify(e))),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
        },
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => TLExpr::Aggregate {
            op: op.clone(),
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
            group_by: group_by.clone(),
        },
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(algebraic_simplify(condition)),
            then_branch: Box::new(algebraic_simplify(then_branch)),
            else_branch: Box::new(algebraic_simplify(else_branch)),
        },
        TLExpr::Let { var, value, body } => TLExpr::Let {
            var: var.clone(),
            value: Box::new(algebraic_simplify(value)),
            body: Box::new(algebraic_simplify(body)),
        },

        // Leaves
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

/// Substitute a variable with a value in an expression (for Let binding propagation)
fn substitute(expr: &TLExpr, var: &str, value: &TLExpr) -> TLExpr {
    match expr {
        // If we find a predicate matching the variable name with no args, substitute
        TLExpr::Pred { name, args } if name == var && args.is_empty() => value.clone(),

        // For predicates with args or different names, keep them
        TLExpr::Pred { .. } => expr.clone(),

        // Recursively substitute in binary operations
        TLExpr::And(l, r) => TLExpr::And(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Imply(l, r) => TLExpr::Imply(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Add(l, r) => TLExpr::Add(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Sub(l, r) => TLExpr::Sub(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Mul(l, r) => TLExpr::Mul(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Div(l, r) => TLExpr::Div(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Pow(l, r) => TLExpr::Pow(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Mod(l, r) => TLExpr::Mod(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Min(l, r) => TLExpr::Min(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Max(l, r) => TLExpr::Max(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Eq(l, r) => TLExpr::Eq(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Lt(l, r) => TLExpr::Lt(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Gt(l, r) => TLExpr::Gt(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Lte(l, r) => TLExpr::Lte(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Gte(l, r) => TLExpr::Gte(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),

        // Recursively substitute in unary operations
        TLExpr::Not(e) => TLExpr::Not(Box::new(substitute(e, var, value))),
        TLExpr::Box(e) => TLExpr::Box(Box::new(substitute(e, var, value))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(substitute(e, var, value))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(substitute(e, var, value))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(substitute(e, var, value))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(substitute(e, var, value))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(substitute(before, var, value)),
            after: Box::new(substitute(after, var, value)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(substitute(left, var, value)),
            right: Box::new(substitute(right, var, value)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(substitute(left, var, value)),
            right: Box::new(substitute(right, var, value)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(substitute(expr, var, value)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(substitute(premise, var, value)),
            conclusion: Box::new(substitute(conclusion, var, value)),
        },

        // Probabilistic operators
        TLExpr::SoftExists {
            var: v,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: v.clone(),
            domain: domain.clone(),
            body: Box::new(if v == var {
                (**body).clone()
            } else {
                substitute(body, var, value)
            }),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var: v,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: v.clone(),
            domain: domain.clone(),
            body: Box::new(if v == var {
                (**body).clone()
            } else {
                substitute(body, var, value)
            }),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(substitute(rule, var, value)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, substitute(e, var, value)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(substitute(released, var, value)),
            releaser: Box::new(substitute(releaser, var, value)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(substitute(before, var, value)),
            after: Box::new(substitute(after, var, value)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(substitute(released, var, value)),
            releaser: Box::new(substitute(releaser, var, value)),
        },

        TLExpr::Score(e) => TLExpr::Score(Box::new(substitute(e, var, value))),
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(substitute(e, var, value))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(substitute(e, var, value))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(substitute(e, var, value))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(substitute(e, var, value))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(substitute(e, var, value))),
        TLExpr::Exp(e) => TLExpr::Exp(Box::new(substitute(e, var, value))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(substitute(e, var, value))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(substitute(e, var, value))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(substitute(e, var, value))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(substitute(e, var, value))),

        // For quantifiers and aggregates, don't substitute if the variable shadows
        TLExpr::Exists {
            var: qvar,
            domain,
            body,
        } => {
            if qvar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::Exists {
                    var: qvar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        TLExpr::ForAll {
            var: qvar,
            domain,
            body,
        } => {
            if qvar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::ForAll {
                    var: qvar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        TLExpr::Aggregate {
            op,
            var: avar,
            domain,
            body,
            group_by,
        } => {
            if avar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::Aggregate {
                    op: op.clone(),
                    var: avar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                    group_by: group_by.clone(),
                }
            }
        }

        // For Let bindings, handle shadowing and substitute recursively
        TLExpr::Let {
            var: lvar,
            value: lvalue,
            body,
        } => {
            let new_value = substitute(lvalue, var, value);
            if lvar == var {
                // Variable is shadowed in body, don't substitute there
                TLExpr::Let {
                    var: lvar.clone(),
                    value: Box::new(new_value),
                    body: body.clone(),
                }
            } else {
                TLExpr::Let {
                    var: lvar.clone(),
                    value: Box::new(new_value),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }

        // For if-then-else, substitute in all branches
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(substitute(condition, var, value)),
            then_branch: Box::new(substitute(then_branch, var, value)),
            else_branch: Box::new(substitute(else_branch, var, value)),
        },

        // Constants remain unchanged
        TLExpr::Constant(_) => expr.clone(),
    }
}

/// Propagate constants through Let bindings
pub fn propagate_constants(expr: &TLExpr) -> TLExpr {
    match expr {
        // If the Let binding value is a constant, substitute it into the body
        TLExpr::Let { var, value, body } => {
            let optimized_value = propagate_constants(value);
            let optimized_body = propagate_constants(body);

            // If the value is constant, substitute it
            if matches!(optimized_value, TLExpr::Constant(_)) {
                substitute(&optimized_body, var, &optimized_value)
            } else {
                TLExpr::Let {
                    var: var.clone(),
                    value: Box::new(optimized_value),
                    body: Box::new(optimized_body),
                }
            }
        }

        // Recursively propagate in other expressions
        TLExpr::And(l, r) => TLExpr::And(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Imply(l, r) => TLExpr::Imply(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Add(l, r) => TLExpr::Add(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Sub(l, r) => TLExpr::Sub(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Mul(l, r) => TLExpr::Mul(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Div(l, r) => TLExpr::Div(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Pow(l, r) => TLExpr::Pow(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Mod(l, r) => TLExpr::Mod(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Min(l, r) => TLExpr::Min(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Max(l, r) => TLExpr::Max(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Eq(l, r) => TLExpr::Eq(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Lt(l, r) => TLExpr::Lt(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Gt(l, r) => TLExpr::Gt(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Lte(l, r) => TLExpr::Lte(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Gte(l, r) => TLExpr::Gte(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Not(e) => TLExpr::Not(Box::new(propagate_constants(e))),
        TLExpr::Score(e) => TLExpr::Score(Box::new(propagate_constants(e))),
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(propagate_constants(e))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(propagate_constants(e))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(propagate_constants(e))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(propagate_constants(e))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(propagate_constants(e))),
        TLExpr::Exp(e) => TLExpr::Exp(Box::new(propagate_constants(e))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(propagate_constants(e))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(propagate_constants(e))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(propagate_constants(e))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(propagate_constants(e))),
        TLExpr::Box(e) => TLExpr::Box(Box::new(propagate_constants(e))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(propagate_constants(e))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(propagate_constants(e))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(propagate_constants(e))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(propagate_constants(e))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(propagate_constants(before)),
            after: Box::new(propagate_constants(after)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(propagate_constants(left)),
            right: Box::new(propagate_constants(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(propagate_constants(left)),
            right: Box::new(propagate_constants(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(propagate_constants(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(propagate_constants(premise)),
            conclusion: Box::new(propagate_constants(conclusion)),
        },

        // Probabilistic operators
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(propagate_constants(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, propagate_constants(e)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(propagate_constants(released)),
            releaser: Box::new(propagate_constants(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(propagate_constants(before)),
            after: Box::new(propagate_constants(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(propagate_constants(released)),
            releaser: Box::new(propagate_constants(releaser)),
        },

        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
        },
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => TLExpr::Aggregate {
            op: op.clone(),
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
            group_by: group_by.clone(),
        },
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(propagate_constants(condition)),
            then_branch: Box::new(propagate_constants(then_branch)),
            else_branch: Box::new(propagate_constants(else_branch)),
        },
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

/// Apply multiple optimization passes in sequence
pub fn optimize_expr(expr: &TLExpr) -> TLExpr {
    // Apply optimizations iteratively until no more changes occur
    // This handles nested Let bindings and cascading optimizations
    let mut current = expr.clone();
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10; // Prevent infinite loops

    loop {
        let propagated = propagate_constants(&current);
        let folded = constant_fold(&propagated);
        let simplified = algebraic_simplify(&folded);

        // If no change occurred, we're done
        if simplified == current || iterations >= MAX_ITERATIONS {
            return simplified;
        }

        current = simplified;
        iterations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_fold_addition() {
        let expr = TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_constant_fold_multiplication() {
        let expr = TLExpr::mul(TLExpr::constant(4.0), TLExpr::constant(5.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_constant_fold_nested() {
        // (2 + 3) * 4 = 20
        let expr = TLExpr::mul(
            TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
            TLExpr::constant(4.0),
        );
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_algebraic_simplify_add_zero() {
        let expr = TLExpr::add(TLExpr::constant(5.0), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_algebraic_simplify_mul_one() {
        let expr = TLExpr::mul(TLExpr::constant(7.0), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(7.0));
    }

    #[test]
    fn test_algebraic_simplify_mul_zero() {
        let expr = TLExpr::mul(TLExpr::constant(7.0), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_algebraic_simplify_double_negation() {
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::constant(5.0)));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_optimize_expr_combined() {
        // (2 + 3) * 1 should become 5
        let expr = TLExpr::mul(
            TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
            TLExpr::constant(1.0),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_constant_fold_trig() {
        let expr = TLExpr::sin(TLExpr::constant(0.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_constant_fold_sqrt() {
        let expr = TLExpr::sqrt(TLExpr::constant(4.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(2.0));
    }

    #[test]
    fn test_algebraic_simplify_power_identities() {
        // x^0 = 1
        let expr = TLExpr::pow(TLExpr::constant(42.0), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));

        // x^1 = x
        let expr2 = TLExpr::pow(TLExpr::constant(42.0), TLExpr::constant(1.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(42.0));
    }

    #[test]
    fn test_let_binding_constant_propagation() {
        // let x = 5 in x + x should become 10
        let expr = TLExpr::let_binding(
            "x",
            TLExpr::constant(5.0),
            TLExpr::add(TLExpr::pred("x", vec![]), TLExpr::pred("x", vec![])),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(10.0));
    }

    #[test]
    fn test_let_binding_nested_propagation() {
        // let x = 3 in (let y = x + 2 in x * y) should become 15
        let expr = TLExpr::let_binding(
            "x",
            TLExpr::constant(3.0),
            TLExpr::let_binding(
                "y",
                TLExpr::add(TLExpr::pred("x", vec![]), TLExpr::constant(2.0)),
                TLExpr::mul(TLExpr::pred("x", vec![]), TLExpr::pred("y", vec![])),
            ),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(15.0));
    }

    #[test]
    fn test_let_binding_shadowing() {
        // let x = 5 in (let x = 3 in x) should become 3
        let expr = TLExpr::let_binding(
            "x",
            TLExpr::constant(5.0),
            TLExpr::let_binding("x", TLExpr::constant(3.0), TLExpr::pred("x", vec![])),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(3.0));
    }

    #[test]
    fn test_let_binding_no_propagation_for_expressions() {
        // let x = y + 1 in x * x should not be fully evaluated
        let expr = TLExpr::let_binding(
            "x",
            TLExpr::add(TLExpr::pred("y", vec![]), TLExpr::constant(1.0)),
            TLExpr::mul(TLExpr::pred("x", vec![]), TLExpr::pred("x", vec![])),
        );
        let optimized = optimize_expr(&expr);
        // Should keep the Let binding since x is not a constant
        assert!(matches!(optimized, TLExpr::Let { .. }));
    }

    #[test]
    fn test_substitute_respects_shadowing_in_quantifiers() {
        // Substitution should not affect shadowed variables in quantifiers
        let expr = TLExpr::exists("x", "Domain", TLExpr::pred("x", vec![]));
        let substituted = substitute(&expr, "x", &TLExpr::constant(5.0));
        // x is shadowed, so it should remain unchanged
        assert_eq!(substituted, expr);
    }

    #[test]
    fn test_propagate_constants_complex() {
        // let a = 2 in (let b = a * 3 in b + a) should become 8
        let expr = TLExpr::let_binding(
            "a",
            TLExpr::constant(2.0),
            TLExpr::let_binding(
                "b",
                TLExpr::mul(TLExpr::pred("a", vec![]), TLExpr::constant(3.0)),
                TLExpr::add(TLExpr::pred("b", vec![]), TLExpr::pred("a", vec![])),
            ),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(8.0));
    }

    #[test]
    fn test_constant_fold_min_max() {
        let expr1 = TLExpr::min(TLExpr::constant(3.0), TLExpr::constant(7.0));
        let folded1 = constant_fold(&expr1);
        assert_eq!(folded1, TLExpr::Constant(3.0));

        let expr2 = TLExpr::max(TLExpr::constant(3.0), TLExpr::constant(7.0));
        let folded2 = constant_fold(&expr2);
        assert_eq!(folded2, TLExpr::Constant(7.0));
    }

    #[test]
    fn test_constant_fold_modulo() {
        let expr = TLExpr::modulo(TLExpr::constant(10.0), TLExpr::constant(3.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(1.0));
    }

    // ===== Advanced Algebraic Simplification Tests =====

    // Logical Laws - AND
    #[test]
    fn test_and_idempotence() {
        // A ∧ A = A
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::and(p.clone(), p.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    #[test]
    fn test_and_identity() {
        // A ∧ TRUE = A
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::and(p.clone(), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    #[test]
    fn test_and_annihilation() {
        // A ∧ FALSE = FALSE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::and(p, TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_and_complement() {
        // A ∧ ¬A = FALSE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::and(p.clone(), TLExpr::negate(p));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_and_absorption() {
        // A ∧ (A ∨ B) = A
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let expr = TLExpr::and(p.clone(), TLExpr::or(p.clone(), q));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    // Logical Laws - OR
    #[test]
    fn test_or_idempotence() {
        // A ∨ A = A
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::or(p.clone(), p.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    #[test]
    fn test_or_identity() {
        // A ∨ FALSE = A
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::or(p.clone(), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    #[test]
    fn test_or_annihilation() {
        // A ∨ TRUE = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::or(p, TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_or_complement() {
        // A ∨ ¬A = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::or(p.clone(), TLExpr::negate(p));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_or_absorption() {
        // A ∨ (A ∧ B) = A
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let expr = TLExpr::or(p.clone(), TLExpr::and(p.clone(), q));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    // Implication Laws
    #[test]
    fn test_imply_true_antecedent() {
        // TRUE → P = P
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::imply(TLExpr::constant(1.0), p.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, p);
    }

    #[test]
    fn test_imply_false_antecedent() {
        // FALSE → P = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::imply(TLExpr::constant(0.0), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_imply_true_consequent() {
        // P → TRUE = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::imply(p, TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_imply_false_consequent() {
        // P → FALSE = ¬P
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::imply(p.clone(), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::negate(p));
    }

    #[test]
    fn test_imply_reflexive() {
        // P → P = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::imply(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    // Comparison Simplifications
    #[test]
    fn test_eq_reflexive() {
        // x = x → TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::eq(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_lt_irreflexive() {
        // x < x → FALSE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::lt(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_gt_irreflexive() {
        // x > x → FALSE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::gt(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_lte_reflexive() {
        // x <= x → TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::lte(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_gte_reflexive() {
        // x >= x → TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::gte(p.clone(), p);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    // Arithmetic Simplifications
    #[test]
    fn test_div_self() {
        // x / x = 1 (for constant x != 0)
        let expr = TLExpr::div(TLExpr::constant(5.0), TLExpr::constant(5.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    // Modal Logic Simplifications
    #[test]
    fn test_box_constant() {
        // □(TRUE) = TRUE, □(FALSE) = FALSE
        let expr1 = TLExpr::modal_box(TLExpr::constant(1.0));
        let simplified1 = algebraic_simplify(&expr1);
        assert_eq!(simplified1, TLExpr::Constant(1.0));

        let expr2 = TLExpr::modal_box(TLExpr::constant(0.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_diamond_constant() {
        // ◇(TRUE) = TRUE, ◇(FALSE) = FALSE
        let expr1 = TLExpr::modal_diamond(TLExpr::constant(1.0));
        let simplified1 = algebraic_simplify(&expr1);
        assert_eq!(simplified1, TLExpr::Constant(1.0));

        let expr2 = TLExpr::modal_diamond(TLExpr::constant(0.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(0.0));
    }

    // Temporal Logic Simplifications
    #[test]
    fn test_next_constant() {
        // X(TRUE) = TRUE, X(FALSE) = FALSE
        let expr1 = TLExpr::next(TLExpr::constant(1.0));
        let simplified1 = algebraic_simplify(&expr1);
        assert_eq!(simplified1, TLExpr::Constant(1.0));

        let expr2 = TLExpr::next(TLExpr::constant(0.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_eventually_constant() {
        // F(TRUE) = TRUE, F(FALSE) = FALSE
        let expr1 = TLExpr::eventually(TLExpr::constant(1.0));
        let simplified1 = algebraic_simplify(&expr1);
        assert_eq!(simplified1, TLExpr::Constant(1.0));

        let expr2 = TLExpr::eventually(TLExpr::constant(0.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_eventually_idempotence() {
        // F(F(P)) = F(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::eventually(TLExpr::eventually(p.clone()));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::eventually(p));
    }

    #[test]
    fn test_always_constant() {
        // G(TRUE) = TRUE, G(FALSE) = FALSE
        let expr1 = TLExpr::always(TLExpr::constant(1.0));
        let simplified1 = algebraic_simplify(&expr1);
        assert_eq!(simplified1, TLExpr::Constant(1.0));

        let expr2 = TLExpr::always(TLExpr::constant(0.0));
        let simplified2 = algebraic_simplify(&expr2);
        assert_eq!(simplified2, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_always_idempotence() {
        // G(G(P)) = G(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::always(TLExpr::always(p.clone()));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::always(p));
    }

    #[test]
    fn test_until_true_consequent() {
        // P U TRUE = TRUE
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::until(p, TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_until_false_antecedent() {
        // FALSE U P = F(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::until(TLExpr::constant(0.0), p.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::eventually(p));
    }

    // Combined Optimization Tests
    #[test]
    fn test_combined_logical_simplification() {
        // (P ∧ TRUE) ∨ FALSE should become P
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::or(
            TLExpr::and(p.clone(), TLExpr::constant(1.0)),
            TLExpr::constant(0.0),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, p);
    }

    #[test]
    fn test_combined_implication_simplification() {
        // (P → TRUE) ∧ (FALSE → Q) should become TRUE
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let expr = TLExpr::and(
            TLExpr::imply(p, TLExpr::constant(1.0)),
            TLExpr::imply(TLExpr::constant(0.0), q),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(1.0));
    }
}
