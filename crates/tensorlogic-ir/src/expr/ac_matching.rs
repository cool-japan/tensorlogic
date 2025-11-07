//! Associative-Commutative (AC) pattern matching for logical expressions.
//!
//! This module provides AC-matching capabilities that recognize equivalent expressions
//! under associativity and commutativity, such as:
//! - `A ∧ B ≡ B ∧ A` (commutativity)
//! - `(A ∧ B) ∧ C ≡ A ∧ (B ∧ C)` (associativity)
//!
//! AC-matching is crucial for advanced rewriting systems where the order and
//! nesting of operators should not affect pattern matching.

use std::collections::HashMap;

use super::TLExpr;

/// Operators that are associative and commutative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ACOperator {
    /// Logical AND (∧)
    And,
    /// Logical OR (∨)
    Or,
    /// Addition (+)
    Add,
    /// Multiplication (*)
    Mul,
    /// Min operation
    Min,
    /// Max operation
    Max,
}

impl ACOperator {
    /// Check if an expression uses this AC operator.
    pub fn matches_expr(&self, expr: &TLExpr) -> bool {
        matches!(
            (self, expr),
            (ACOperator::And, TLExpr::And(_, _))
                | (ACOperator::Or, TLExpr::Or(_, _))
                | (ACOperator::Add, TLExpr::Add(_, _))
                | (ACOperator::Mul, TLExpr::Mul(_, _))
                | (ACOperator::Min, TLExpr::Min(_, _))
                | (ACOperator::Max, TLExpr::Max(_, _))
        )
    }

    /// Extract operands from an AC expression.
    pub fn extract_operands<'a>(&self, expr: &'a TLExpr) -> Option<(&'a TLExpr, &'a TLExpr)> {
        match (self, expr) {
            (ACOperator::And, TLExpr::And(l, r)) => Some((l, r)),
            (ACOperator::Or, TLExpr::Or(l, r)) => Some((l, r)),
            (ACOperator::Add, TLExpr::Add(l, r)) => Some((l, r)),
            (ACOperator::Mul, TLExpr::Mul(l, r)) => Some((l, r)),
            (ACOperator::Min, TLExpr::Min(l, r)) => Some((l, r)),
            (ACOperator::Max, TLExpr::Max(l, r)) => Some((l, r)),
            _ => None,
        }
    }
}

/// Flatten an AC expression into a list of operands.
///
/// For example, `(A ∧ B) ∧ (C ∧ D)` becomes `[A, B, C, D]`.
pub fn flatten_ac(expr: &TLExpr, op: ACOperator) -> Vec<TLExpr> {
    let mut result = Vec::new();
    flatten_ac_recursive(expr, op, &mut result);
    result
}

fn flatten_ac_recursive(expr: &TLExpr, op: ACOperator, acc: &mut Vec<TLExpr>) {
    if let Some((left, right)) = op.extract_operands(expr) {
        flatten_ac_recursive(left, op, acc);
        flatten_ac_recursive(right, op, acc);
    } else {
        acc.push(expr.clone());
    }
}

/// Normalize an AC expression by sorting operands.
///
/// This creates a canonical form for AC expressions, making them easier to compare.
pub fn normalize_ac(expr: &TLExpr, op: ACOperator) -> TLExpr {
    if !op.matches_expr(expr) {
        return expr.clone();
    }

    let mut operands = flatten_ac(expr, op);

    // Sort operands by their debug representation (simple but effective)
    operands.sort_by_cached_key(|e| format!("{:?}", e));

    // Rebuild the expression
    if operands.is_empty() {
        return expr.clone();
    }

    let mut result = operands.pop().unwrap();
    while let Some(operand) = operands.pop() {
        result = match op {
            ACOperator::And => TLExpr::and(operand, result),
            ACOperator::Or => TLExpr::or(operand, result),
            ACOperator::Add => TLExpr::add(operand, result),
            ACOperator::Mul => TLExpr::mul(operand, result),
            ACOperator::Min => TLExpr::min(operand, result),
            ACOperator::Max => TLExpr::max(operand, result),
        };
    }

    result
}

/// Check if two expressions are AC-equivalent.
///
/// This recursively normalizes both expressions and compares them.
pub fn ac_equivalent(expr1: &TLExpr, expr2: &TLExpr) -> bool {
    // Try each AC operator
    for op in &[
        ACOperator::And,
        ACOperator::Or,
        ACOperator::Add,
        ACOperator::Mul,
        ACOperator::Min,
        ACOperator::Max,
    ] {
        if op.matches_expr(expr1) || op.matches_expr(expr2) {
            let norm1 = normalize_ac(expr1, *op);
            let norm2 = normalize_ac(expr2, *op);
            return norm1 == norm2;
        }
    }

    // If neither is an AC operator, just compare directly
    expr1 == expr2
}

/// AC pattern matching with variable bindings.
///
/// This is more sophisticated than simple AC-equivalence checking, as it allows
/// pattern variables to match subsets of operands.
#[derive(Debug, Clone)]
pub struct ACPattern {
    /// The AC operator for this pattern
    pub operator: ACOperator,
    /// Fixed operands that must match exactly
    pub fixed_operands: Vec<TLExpr>,
    /// Variable operands that can match multiple elements
    pub variable_operands: Vec<String>,
}

impl ACPattern {
    /// Create a new AC pattern.
    pub fn new(operator: ACOperator) -> Self {
        Self {
            operator,
            fixed_operands: Vec::new(),
            variable_operands: Vec::new(),
        }
    }

    /// Add a fixed operand to the pattern.
    pub fn with_fixed(mut self, operand: TLExpr) -> Self {
        self.fixed_operands.push(operand);
        self
    }

    /// Add a variable operand to the pattern.
    pub fn with_variable(mut self, var: impl Into<String>) -> Self {
        self.variable_operands.push(var.into());
        self
    }

    /// Try to match this pattern against an expression.
    ///
    /// Returns bindings for variable operands if successful.
    pub fn matches(&self, expr: &TLExpr) -> Option<HashMap<String, Vec<TLExpr>>> {
        // Extract operands from expression
        let expr_operands = flatten_ac(expr, self.operator);

        // Check if all fixed operands are present
        let mut remaining = expr_operands.clone();
        for fixed in &self.fixed_operands {
            if let Some(pos) = remaining.iter().position(|e| e == fixed) {
                remaining.remove(pos);
            } else {
                return None; // Fixed operand not found
            }
        }

        // If we have no variable operands, remaining should be empty
        if self.variable_operands.is_empty() {
            if remaining.is_empty() {
                return Some(HashMap::new());
            } else {
                return None;
            }
        }

        // For single variable operand, it matches all remaining
        if self.variable_operands.len() == 1 {
            let mut bindings = HashMap::new();
            bindings.insert(self.variable_operands[0].clone(), remaining);
            return Some(bindings);
        }

        // For multiple variable operands, we need to find all partitions
        // This is NP-complete in general, so we use a simple heuristic:
        // distribute remaining operands evenly
        if remaining.len() < self.variable_operands.len() {
            return None; // Not enough operands
        }

        let mut bindings = HashMap::new();
        let chunk_size = remaining.len() / self.variable_operands.len();
        let mut start = 0;

        for (i, var) in self.variable_operands.iter().enumerate() {
            let end = if i == self.variable_operands.len() - 1 {
                remaining.len() // Last variable gets all remaining
            } else {
                start + chunk_size
            };

            let chunk = remaining[start..end].to_vec();
            bindings.insert(var.clone(), chunk);
            start = end;
        }

        Some(bindings)
    }
}

/// Multiset for AC matching.
///
/// Represents a collection of elements where order doesn't matter but multiplicity does.
#[derive(Debug, Clone)]
pub struct Multiset<T> {
    elements: HashMap<T, usize>,
}

impl<T: Eq + std::hash::Hash + Clone> Multiset<T> {
    /// Create an empty multiset.
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
        }
    }

    /// Create a multiset from a vector.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut multiset = Self::new();
        for elem in vec {
            multiset.insert(elem);
        }
        multiset
    }

    /// Insert an element into the multiset.
    pub fn insert(&mut self, elem: T) {
        *self.elements.entry(elem).or_insert(0) += 1;
    }

    /// Remove an element from the multiset.
    pub fn remove(&mut self, elem: &T) -> bool {
        if let Some(count) = self.elements.get_mut(elem) {
            if *count > 0 {
                *count -= 1;
                if *count == 0 {
                    self.elements.remove(elem);
                }
                return true;
            }
        }
        false
    }

    /// Check if the multiset contains an element.
    pub fn contains(&self, elem: &T) -> bool {
        self.elements.get(elem).is_some_and(|&count| count > 0)
    }

    /// Check if the multiset is empty.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get the number of occurrences of an element.
    pub fn count(&self, elem: &T) -> usize {
        self.elements.get(elem).copied().unwrap_or(0)
    }

    /// Check if this is a subset of another multiset.
    pub fn is_subset(&self, other: &Multiset<T>) -> bool {
        for (elem, count) in &self.elements {
            if other.count(elem) < *count {
                return false;
            }
        }
        true
    }
}

impl<T: Eq + std::hash::Hash + Clone> Default for Multiset<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + std::hash::Hash> PartialEq for Multiset<T> {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

impl<T: Eq + std::hash::Hash> Eq for Multiset<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_flatten_ac_and() {
        // (A ∧ B) ∧ C should flatten to [A, B, C]
        let expr = TLExpr::and(
            TLExpr::and(
                TLExpr::pred("A", vec![Term::var("x")]),
                TLExpr::pred("B", vec![Term::var("y")]),
            ),
            TLExpr::pred("C", vec![Term::var("z")]),
        );

        let operands = flatten_ac(&expr, ACOperator::And);
        assert_eq!(operands.len(), 3);
    }

    #[test]
    fn test_normalize_ac() {
        // B ∧ A should normalize to A ∧ B
        let expr1 = TLExpr::and(
            TLExpr::pred("B", vec![Term::var("y")]),
            TLExpr::pred("A", vec![Term::var("x")]),
        );

        let expr2 = TLExpr::and(
            TLExpr::pred("A", vec![Term::var("x")]),
            TLExpr::pred("B", vec![Term::var("y")]),
        );

        let norm1 = normalize_ac(&expr1, ACOperator::And);
        let norm2 = normalize_ac(&expr2, ACOperator::And);

        assert_eq!(norm1, norm2);
    }

    #[test]
    fn test_ac_equivalent() {
        // (A ∧ B) ∧ C ≡ C ∧ (B ∧ A)
        let expr1 = TLExpr::and(
            TLExpr::and(
                TLExpr::pred("A", vec![Term::var("x")]),
                TLExpr::pred("B", vec![Term::var("y")]),
            ),
            TLExpr::pred("C", vec![Term::var("z")]),
        );

        let expr2 = TLExpr::and(
            TLExpr::pred("C", vec![Term::var("z")]),
            TLExpr::and(
                TLExpr::pred("B", vec![Term::var("y")]),
                TLExpr::pred("A", vec![Term::var("x")]),
            ),
        );

        assert!(ac_equivalent(&expr1, &expr2));
    }

    #[test]
    fn test_ac_pattern_simple() {
        // Pattern: A ∧ <var>
        let pattern = ACPattern::new(ACOperator::And)
            .with_fixed(TLExpr::pred("A", vec![Term::var("x")]))
            .with_variable("rest");

        // Expression: A ∧ B ∧ C
        let expr = TLExpr::and(
            TLExpr::and(
                TLExpr::pred("A", vec![Term::var("x")]),
                TLExpr::pred("B", vec![Term::var("y")]),
            ),
            TLExpr::pred("C", vec![Term::var("z")]),
        );

        let bindings = pattern.matches(&expr).unwrap();
        assert!(bindings.contains_key("rest"));
        assert_eq!(bindings.get("rest").unwrap().len(), 2); // B and C
    }

    #[test]
    fn test_multiset_operations() {
        let mut ms1 = Multiset::new();
        ms1.insert("A");
        ms1.insert("B");
        ms1.insert("A"); // A appears twice

        assert_eq!(ms1.count(&"A"), 2);
        assert_eq!(ms1.count(&"B"), 1);
        assert!(ms1.contains(&"A"));

        let mut ms2 = Multiset::new();
        ms2.insert("A");

        assert!(ms2.is_subset(&ms1));
        assert!(!ms1.is_subset(&ms2));
    }

    #[test]
    fn test_multiset_equality() {
        let ms1 = Multiset::from_vec(vec!["A", "B", "A"]);
        let ms2 = Multiset::from_vec(vec!["B", "A", "A"]);
        let ms3 = Multiset::from_vec(vec!["A", "B"]);

        assert_eq!(ms1, ms2); // Order doesn't matter
        assert_ne!(ms1, ms3); // Multiplicity matters
    }
}
