//! Expression complexity analysis.
//!
//! This module provides analysis tools for estimating the computational
//! complexity of TLExpr expressions. It can estimate:
//!
//! - Operation counts (additions, multiplications, etc.)
//! - Computational cost (weighted by operation type)
//! - Nesting depth
//! - Memory footprint estimates
//!
//! # Usage
//!
//! ```
//! use tensorlogic_compiler::optimize::{analyze_complexity, ExpressionComplexity};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let x = TLExpr::pred("x", vec![Term::var("i")]);
//! let expr = TLExpr::mul(TLExpr::add(x, TLExpr::Constant(1.0)), TLExpr::Constant(2.0));
//! let complexity = analyze_complexity(&expr);
//!
//! println!("Total cost: {}", complexity.total_cost());
//! println!("Operations: {}", complexity.total_operations());
//! ```

use tensorlogic_ir::TLExpr;

/// Cost weights for different operation types.
#[derive(Debug, Clone)]
pub struct CostWeights {
    /// Cost of addition/subtraction
    pub add_sub: f64,
    /// Cost of multiplication
    pub mul: f64,
    /// Cost of division
    pub div: f64,
    /// Cost of power operation
    pub pow: f64,
    /// Cost of exponential
    pub exp: f64,
    /// Cost of logarithm
    pub log: f64,
    /// Cost of square root
    pub sqrt: f64,
    /// Cost of absolute value
    pub abs: f64,
    /// Cost of comparison
    pub cmp: f64,
    /// Cost of reduction (per dimension)
    pub reduction: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            add_sub: 1.0,
            mul: 2.0,
            div: 4.0,
            pow: 8.0,
            exp: 10.0,
            log: 10.0,
            sqrt: 4.0,
            abs: 1.0,
            cmp: 1.0,
            reduction: 5.0,
        }
    }
}

impl CostWeights {
    /// Weights optimized for GPU execution where parallelism matters more.
    pub fn gpu_optimized() -> Self {
        Self {
            add_sub: 1.0,
            mul: 1.0,
            div: 2.0,
            pow: 4.0,
            exp: 3.0,
            log: 3.0,
            sqrt: 2.0,
            abs: 1.0,
            cmp: 1.0,
            reduction: 10.0, // Reductions are expensive on GPU
        }
    }

    /// Weights optimized for SIMD execution.
    pub fn simd_optimized() -> Self {
        Self {
            add_sub: 1.0,
            mul: 1.0,
            div: 3.0,
            pow: 6.0,
            exp: 8.0,
            log: 8.0,
            sqrt: 3.0,
            abs: 1.0,
            cmp: 1.0,
            reduction: 3.0,
        }
    }
}

/// Detailed complexity analysis of an expression.
#[derive(Debug, Clone, Default)]
pub struct ExpressionComplexity {
    /// Number of addition operations
    pub additions: usize,
    /// Number of subtraction operations
    pub subtractions: usize,
    /// Number of multiplication operations
    pub multiplications: usize,
    /// Number of division operations
    pub divisions: usize,
    /// Number of power operations
    pub powers: usize,
    /// Number of exponential operations
    pub exponentials: usize,
    /// Number of logarithm operations
    pub logarithms: usize,
    /// Number of square root operations
    pub square_roots: usize,
    /// Number of absolute value operations
    pub absolute_values: usize,
    /// Number of negation operations
    pub negations: usize,
    /// Number of comparison operations
    pub comparisons: usize,
    /// Number of logical AND operations
    pub logical_ands: usize,
    /// Number of logical OR operations
    pub logical_ors: usize,
    /// Number of logical NOT operations
    pub logical_nots: usize,
    /// Number of existential quantifiers
    pub existential_quantifiers: usize,
    /// Number of universal quantifiers
    pub universal_quantifiers: usize,
    /// Number of conditional expressions
    pub conditionals: usize,
    /// Number of predicate applications
    pub predicates: usize,
    /// Number of constants
    pub constants: usize,
    /// Number of variables
    pub variables: usize,
    /// Number of min operations
    pub min_operations: usize,
    /// Number of max operations
    pub max_operations: usize,
    /// Maximum nesting depth
    pub max_depth: usize,
    /// Number of unique variable names
    pub unique_variables: usize,
    /// Number of unique predicate names
    pub unique_predicates: usize,
}

impl ExpressionComplexity {
    /// Get total number of arithmetic operations.
    pub fn arithmetic_operations(&self) -> usize {
        self.additions
            + self.subtractions
            + self.multiplications
            + self.divisions
            + self.powers
            + self.exponentials
            + self.logarithms
            + self.square_roots
            + self.absolute_values
            + self.negations
    }

    /// Get total number of logical operations.
    pub fn logical_operations(&self) -> usize {
        self.logical_ands + self.logical_ors + self.logical_nots
    }

    /// Get total number of operations.
    pub fn total_operations(&self) -> usize {
        self.arithmetic_operations()
            + self.logical_operations()
            + self.comparisons
            + self.conditionals
            + self.min_operations
            + self.max_operations
    }

    /// Calculate total weighted cost using default weights.
    pub fn total_cost(&self) -> f64 {
        self.total_cost_with_weights(&CostWeights::default())
    }

    /// Calculate total weighted cost using custom weights.
    pub fn total_cost_with_weights(&self, weights: &CostWeights) -> f64 {
        let mut cost = 0.0;
        cost += (self.additions + self.subtractions) as f64 * weights.add_sub;
        cost += self.multiplications as f64 * weights.mul;
        cost += self.divisions as f64 * weights.div;
        cost += self.powers as f64 * weights.pow;
        cost += self.exponentials as f64 * weights.exp;
        cost += self.logarithms as f64 * weights.log;
        cost += self.square_roots as f64 * weights.sqrt;
        cost += self.absolute_values as f64 * weights.abs;
        cost += self.comparisons as f64 * weights.cmp;
        cost +=
            (self.existential_quantifiers + self.universal_quantifiers) as f64 * weights.reduction;
        cost += self.min_operations as f64 * weights.cmp;
        cost += self.max_operations as f64 * weights.cmp;
        cost
    }

    /// Get the number of leaf nodes (constants, variables, predicates).
    pub fn leaf_count(&self) -> usize {
        self.constants + self.variables + self.predicates
    }

    /// Estimate if this expression would benefit from CSE.
    pub fn cse_potential(&self) -> bool {
        // Heuristic: complex expressions with many operations benefit from CSE
        self.total_operations() > 5 && self.max_depth > 3
    }

    /// Estimate if this expression would benefit from strength reduction.
    pub fn strength_reduction_potential(&self) -> bool {
        self.powers > 0 || self.divisions > 2 || self.exponentials + self.logarithms > 0
    }

    /// Estimate relative complexity as a string descriptor.
    pub fn complexity_level(&self) -> &'static str {
        let total = self.total_operations();
        if total <= 3 {
            "trivial"
        } else if total <= 10 {
            "simple"
        } else if total <= 30 {
            "moderate"
        } else if total <= 100 {
            "complex"
        } else {
            "very_complex"
        }
    }
}

impl std::fmt::Display for ExpressionComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Expression Complexity Analysis:")?;
        writeln!(f, "  Total operations: {}", self.total_operations())?;
        writeln!(
            f,
            "  Arithmetic operations: {}",
            self.arithmetic_operations()
        )?;
        writeln!(f, "  Logical operations: {}", self.logical_operations())?;
        writeln!(f, "  Maximum depth: {}", self.max_depth)?;
        writeln!(f, "  Estimated cost: {:.2}", self.total_cost())?;
        writeln!(f, "  Complexity level: {}", self.complexity_level())?;
        Ok(())
    }
}

/// Analyze the complexity of an expression.
///
/// # Arguments
///
/// * `expr` - The expression to analyze
///
/// # Returns
///
/// A detailed complexity analysis
pub fn analyze_complexity(expr: &TLExpr) -> ExpressionComplexity {
    let mut complexity = ExpressionComplexity::default();
    let mut var_names = std::collections::HashSet::new();
    let mut pred_names = std::collections::HashSet::new();

    analyze_complexity_impl(expr, &mut complexity, 0, &mut var_names, &mut pred_names);

    complexity.unique_variables = var_names.len();
    complexity.unique_predicates = pred_names.len();

    complexity
}

fn analyze_complexity_impl(
    expr: &TLExpr,
    complexity: &mut ExpressionComplexity,
    depth: usize,
    var_names: &mut std::collections::HashSet<String>,
    pred_names: &mut std::collections::HashSet<String>,
) {
    complexity.max_depth = complexity.max_depth.max(depth);

    match expr {
        TLExpr::Add(lhs, rhs) => {
            complexity.additions += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Sub(lhs, rhs) => {
            complexity.subtractions += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Mul(lhs, rhs) => {
            complexity.multiplications += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Div(lhs, rhs) => {
            complexity.divisions += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Pow(base, exp) => {
            complexity.powers += 1;
            analyze_complexity_impl(base, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(exp, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Exp(inner) => {
            complexity.exponentials += 1;
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Log(inner) => {
            complexity.logarithms += 1;
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Sqrt(inner) => {
            complexity.square_roots += 1;
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Abs(inner) => {
            complexity.absolute_values += 1;
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::And(lhs, rhs) => {
            complexity.logical_ands += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Or(lhs, rhs) => {
            complexity.logical_ors += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Not(inner) => {
            complexity.logical_nots += 1;
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Imply(lhs, rhs) => {
            // Implication typically compiles to NOT/OR or other ops
            complexity.logical_nots += 1;
            complexity.logical_ors += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Eq(lhs, rhs)
        | TLExpr::Lt(lhs, rhs)
        | TLExpr::Lte(lhs, rhs)
        | TLExpr::Gt(lhs, rhs)
        | TLExpr::Gte(lhs, rhs) => {
            complexity.comparisons += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Min(lhs, rhs) => {
            complexity.min_operations += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Max(lhs, rhs) => {
            complexity.max_operations += 1;
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Exists { var, body, .. } => {
            complexity.existential_quantifiers += 1;
            var_names.insert(var.clone());
            analyze_complexity_impl(body, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::ForAll { var, body, .. } => {
            complexity.universal_quantifiers += 1;
            var_names.insert(var.clone());
            analyze_complexity_impl(body, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Let {
            var, value, body, ..
        } => {
            var_names.insert(var.clone());
            analyze_complexity_impl(value, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(body, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            complexity.conditionals += 1;
            analyze_complexity_impl(condition, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(then_branch, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(else_branch, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Pred { name, args } => {
            complexity.predicates += 1;
            pred_names.insert(name.clone());
            for arg in args {
                if let tensorlogic_ir::Term::Var(v) = arg {
                    var_names.insert(v.clone());
                }
            }
        }

        TLExpr::Constant(_) => {
            complexity.constants += 1;
        }

        // Modal logic
        TLExpr::Box(inner) | TLExpr::Diamond(inner) => {
            complexity.universal_quantifiers += 1; // Treat as reduction
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        // Temporal logic
        TLExpr::Next(inner) | TLExpr::Eventually(inner) | TLExpr::Always(inner) => {
            complexity.existential_quantifiers += 1; // Treat as reduction
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Until { before, after } => {
            complexity.existential_quantifiers += 1;
            analyze_complexity_impl(before, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(after, complexity, depth + 1, var_names, pred_names);
        }

        // Other variants: minimal complexity impact
        TLExpr::Score(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::FuzzyNot { expr: inner, .. } => {
            analyze_complexity_impl(inner, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Mod(lhs, rhs)
        | TLExpr::TNorm {
            left: lhs,
            right: rhs,
            ..
        }
        | TLExpr::TCoNorm {
            left: lhs,
            right: rhs,
            ..
        }
        | TLExpr::FuzzyImplication {
            premise: lhs,
            conclusion: rhs,
            ..
        }
        | TLExpr::Release {
            released: lhs,
            releaser: rhs,
        }
        | TLExpr::WeakUntil {
            before: lhs,
            after: rhs,
        }
        | TLExpr::StrongRelease {
            released: lhs,
            releaser: rhs,
        } => {
            analyze_complexity_impl(lhs, complexity, depth + 1, var_names, pred_names);
            analyze_complexity_impl(rhs, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::Aggregate { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::WeightedRule { rule: body, .. } => {
            complexity.existential_quantifiers += 1;
            analyze_complexity_impl(body, complexity, depth + 1, var_names, pred_names);
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, expr) in alternatives {
                analyze_complexity_impl(expr, complexity, depth + 1, var_names, pred_names);
            }
        }

        // All other expression types (enhancements)
        _ => {}
    }
}

/// Compare two expressions by their complexity.
///
/// Returns the simpler expression first.
pub fn compare_complexity(expr1: &TLExpr, expr2: &TLExpr) -> std::cmp::Ordering {
    let c1 = analyze_complexity(expr1);
    let c2 = analyze_complexity(expr2);

    let cost1 = c1.total_cost();
    let cost2 = c2.total_cost();

    cost1
        .partial_cmp(&cost2)
        .unwrap_or(std::cmp::Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_simple_addition() {
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let complexity = analyze_complexity(&expr);

        assert_eq!(complexity.additions, 1);
        assert_eq!(complexity.constants, 2);
        assert_eq!(complexity.total_operations(), 1);
    }

    #[test]
    fn test_nested_operations() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::mul(
            TLExpr::add(x.clone(), TLExpr::Constant(1.0)),
            TLExpr::sub(x, TLExpr::Constant(2.0)),
        );
        let complexity = analyze_complexity(&expr);

        assert_eq!(complexity.additions, 1);
        assert_eq!(complexity.subtractions, 1);
        assert_eq!(complexity.multiplications, 1);
        assert_eq!(complexity.predicates, 2);
        assert_eq!(complexity.constants, 2);
    }

    #[test]
    fn test_logical_operations() {
        let a = TLExpr::pred("a", vec![Term::var("x")]);
        let b = TLExpr::pred("b", vec![Term::var("y")]);
        let expr = TLExpr::and(a, TLExpr::negate(b));
        let complexity = analyze_complexity(&expr);

        assert_eq!(complexity.logical_ands, 1);
        assert_eq!(complexity.logical_nots, 1);
        assert_eq!(complexity.predicates, 2);
    }

    #[test]
    fn test_quantifiers() {
        let pred = TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]);
        let expr = TLExpr::exists("x", "D1", TLExpr::forall("y", "D2", pred));
        let complexity = analyze_complexity(&expr);

        assert_eq!(complexity.existential_quantifiers, 1);
        assert_eq!(complexity.universal_quantifiers, 1);
        assert_eq!(complexity.predicates, 1);
        assert_eq!(complexity.unique_variables, 2);
    }

    #[test]
    fn test_depth_calculation() {
        // Depth 3: add -> mul -> pred
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::add(TLExpr::mul(x, TLExpr::Constant(2.0)), TLExpr::Constant(3.0));
        let complexity = analyze_complexity(&expr);

        assert_eq!(complexity.max_depth, 2);
    }

    #[test]
    fn test_cost_calculation() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        // Cost: 1 add + 1 mul = 1*1 + 1*2 = 3
        let expr = TLExpr::add(TLExpr::mul(x, TLExpr::Constant(2.0)), TLExpr::Constant(3.0));
        let complexity = analyze_complexity(&expr);

        let cost = complexity.total_cost();
        assert!(cost > 0.0);
        // 1 add (cost 1) + 1 mul (cost 2) = 3
        assert_eq!(cost, 3.0);
    }

    #[test]
    fn test_gpu_weights() {
        let weights = CostWeights::gpu_optimized();
        assert!(weights.reduction > weights.mul);
    }

    #[test]
    fn test_complexity_level() {
        let simple = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let complex = {
            let mut expr = TLExpr::pred("x", vec![Term::var("i")]);
            for _ in 0..20 {
                expr = TLExpr::add(expr, TLExpr::Constant(1.0));
            }
            expr
        };

        let simple_c = analyze_complexity(&simple);
        let complex_c = analyze_complexity(&complex);

        assert_eq!(simple_c.complexity_level(), "trivial");
        assert!(
            complex_c.complexity_level() == "moderate" || complex_c.complexity_level() == "complex"
        );
    }

    #[test]
    fn test_cse_potential() {
        // Simple expression: no CSE potential
        let simple = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let simple_c = analyze_complexity(&simple);
        assert!(!simple_c.cse_potential());

        // Complex expression: has CSE potential
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let complex = TLExpr::mul(
            TLExpr::exp(TLExpr::add(
                TLExpr::mul(x.clone(), TLExpr::Constant(2.0)),
                TLExpr::Constant(1.0),
            )),
            TLExpr::log(TLExpr::sub(
                TLExpr::div(x, TLExpr::Constant(3.0)),
                TLExpr::Constant(4.0),
            )),
        );
        let complex_c = analyze_complexity(&complex);
        assert!(complex_c.cse_potential());
    }

    #[test]
    fn test_strength_reduction_potential() {
        // Has power: yes
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
        let c = analyze_complexity(&expr);
        assert!(c.strength_reduction_potential());

        // Simple add: no
        let simple = TLExpr::add(x, TLExpr::Constant(1.0));
        let simple_c = analyze_complexity(&simple);
        assert!(!simple_c.strength_reduction_potential());
    }

    #[test]
    fn test_compare_complexity() {
        let simple = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let complex = TLExpr::mul(
            TLExpr::add(x.clone(), TLExpr::Constant(1.0)),
            TLExpr::sub(x, TLExpr::Constant(2.0)),
        );

        let ordering = compare_complexity(&simple, &complex);
        assert_eq!(ordering, std::cmp::Ordering::Less);
    }

    #[test]
    fn test_display() {
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let complexity = analyze_complexity(&expr);
        let display = format!("{}", complexity);

        assert!(display.contains("Expression Complexity Analysis"));
        assert!(display.contains("Total operations:"));
    }

    #[test]
    fn test_arithmetic_vs_logical() {
        let arith = TLExpr::mul(
            TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
            TLExpr::Constant(3.0),
        );
        let logic = TLExpr::and(
            TLExpr::or(TLExpr::pred("a", vec![]), TLExpr::pred("b", vec![])),
            TLExpr::pred("c", vec![]),
        );

        let arith_c = analyze_complexity(&arith);
        let logic_c = analyze_complexity(&logic);

        assert!(arith_c.arithmetic_operations() > 0);
        assert_eq!(arith_c.logical_operations(), 0);
        assert_eq!(logic_c.arithmetic_operations(), 0);
        assert!(logic_c.logical_operations() > 0);
    }

    #[test]
    fn test_unique_variables() {
        let expr = TLExpr::exists(
            "x",
            "D",
            TLExpr::forall(
                "y",
                "D",
                TLExpr::pred("p", vec![Term::var("x"), Term::var("y"), Term::var("z")]),
            ),
        );
        let c = analyze_complexity(&expr);

        assert_eq!(c.unique_variables, 3); // x, y, z
    }

    #[test]
    fn test_unique_predicates() {
        let expr = TLExpr::and(
            TLExpr::pred("foo", vec![Term::var("x")]),
            TLExpr::or(
                TLExpr::pred("bar", vec![Term::var("y")]),
                TLExpr::pred("foo", vec![Term::var("z")]), // Same predicate again
            ),
        );
        let c = analyze_complexity(&expr);

        assert_eq!(c.unique_predicates, 2); // foo, bar
        assert_eq!(c.predicates, 3); // 3 predicate applications
    }
}
