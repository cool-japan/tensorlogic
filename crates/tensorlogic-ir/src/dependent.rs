//! Dependent type system for value-dependent types in TensorLogic.
//!
//! This module implements dependent types, where types can depend on runtime values.
//! This is crucial for tensor operations where dimensions are first-class values.
//!
//! # Examples
//!
//! ```
//! use tensorlogic_ir::dependent::{DependentType, IndexExpr, DimConstraint};
//!
//! // Vector of length n: Vec<n, T>
//! let n = IndexExpr::var("n");
//! let vec_n_int = DependentType::vector(n.clone(), "Int");
//!
//! // Matrix with dimensions m×n: Matrix<m, n, T>
//! let m = IndexExpr::var("m");
//! let matrix_type = DependentType::matrix(m.clone(), n.clone(), "Float");
//!
//! // Bounded vector: Vec<n, T> where n <= 100
//! let constraint = DimConstraint::lte(n.clone(), IndexExpr::constant(100));
//! ```
//!
//! # Key Features
//!
//! - **Index expressions**: Arithmetic on dimension variables
//! - **Dependent function types**: (x: T) -> U(x)
//! - **Refinement types**: Types with predicates on values
//! - **Dimension constraints**: Bounds and relationships between dimensions
//! - **Type-level computation**: Compute types from values

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::{ParametricType, Term};

/// Index expression for dimension calculations.
///
/// Index expressions represent compile-time or runtime values used in type indices,
/// particularly for tensor dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexExpr {
    /// Variable index (e.g., n, m)
    Var(String),
    /// Constant index value
    Const(i64),
    /// Addition: e1 + e2
    Add(Box<IndexExpr>, Box<IndexExpr>),
    /// Subtraction: e1 - e2
    Sub(Box<IndexExpr>, Box<IndexExpr>),
    /// Multiplication: e1 * e2
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    /// Division: e1 / e2
    Div(Box<IndexExpr>, Box<IndexExpr>),
    /// Minimum: min(e1, e2)
    Min(Box<IndexExpr>, Box<IndexExpr>),
    /// Maximum: max(e1, e2)
    Max(Box<IndexExpr>, Box<IndexExpr>),
}

impl IndexExpr {
    /// Create a variable index expression
    pub fn var(name: impl Into<String>) -> Self {
        IndexExpr::Var(name.into())
    }

    /// Create a constant index expression
    pub fn constant(value: i64) -> Self {
        IndexExpr::Const(value)
    }

    /// Addition
    #[allow(clippy::should_implement_trait)]
    pub fn add(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Add(Box::new(left), Box::new(right))
    }

    /// Subtraction
    #[allow(clippy::should_implement_trait)]
    pub fn sub(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Sub(Box::new(left), Box::new(right))
    }

    /// Multiplication
    #[allow(clippy::should_implement_trait)]
    pub fn mul(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Mul(Box::new(left), Box::new(right))
    }

    /// Division
    #[allow(clippy::should_implement_trait)]
    pub fn div(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Div(Box::new(left), Box::new(right))
    }

    /// Minimum
    pub fn min(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Min(Box::new(left), Box::new(right))
    }

    /// Maximum
    pub fn max(left: IndexExpr, right: IndexExpr) -> Self {
        IndexExpr::Max(Box::new(left), Box::new(right))
    }

    /// Get all free variables in this expression
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            IndexExpr::Var(name) => {
                vars.insert(name.clone());
            }
            IndexExpr::Const(_) => {}
            IndexExpr::Add(l, r)
            | IndexExpr::Sub(l, r)
            | IndexExpr::Mul(l, r)
            | IndexExpr::Div(l, r)
            | IndexExpr::Min(l, r)
            | IndexExpr::Max(l, r) => {
                l.collect_vars(vars);
                r.collect_vars(vars);
            }
        }
    }

    /// Substitute variables with index expressions
    pub fn substitute(&self, subst: &HashMap<String, IndexExpr>) -> IndexExpr {
        match self {
            IndexExpr::Var(name) => subst.get(name).cloned().unwrap_or_else(|| self.clone()),
            IndexExpr::Const(_) => self.clone(),
            IndexExpr::Add(l, r) => {
                IndexExpr::Add(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
            IndexExpr::Sub(l, r) => {
                IndexExpr::Sub(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
            IndexExpr::Mul(l, r) => {
                IndexExpr::Mul(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
            IndexExpr::Div(l, r) => {
                IndexExpr::Div(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
            IndexExpr::Min(l, r) => {
                IndexExpr::Min(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
            IndexExpr::Max(l, r) => {
                IndexExpr::Max(Box::new(l.substitute(subst)), Box::new(r.substitute(subst)))
            }
        }
    }

    /// Simplify the index expression
    pub fn simplify(&self) -> IndexExpr {
        match self {
            IndexExpr::Add(l, r) => match (l.simplify(), r.simplify()) {
                (IndexExpr::Const(0), e) | (e, IndexExpr::Const(0)) => e,
                (IndexExpr::Const(a), IndexExpr::Const(b)) => IndexExpr::Const(a + b),
                (l, r) => IndexExpr::Add(Box::new(l), Box::new(r)),
            },
            IndexExpr::Sub(l, r) => match (l.simplify(), r.simplify()) {
                (e, IndexExpr::Const(0)) => e,
                (IndexExpr::Const(a), IndexExpr::Const(b)) => IndexExpr::Const(a - b),
                (l, r) if l == r => IndexExpr::Const(0),
                (l, r) => IndexExpr::Sub(Box::new(l), Box::new(r)),
            },
            IndexExpr::Mul(l, r) => match (l.simplify(), r.simplify()) {
                (IndexExpr::Const(0), _) | (_, IndexExpr::Const(0)) => IndexExpr::Const(0),
                (IndexExpr::Const(1), e) | (e, IndexExpr::Const(1)) => e,
                (IndexExpr::Const(a), IndexExpr::Const(b)) => IndexExpr::Const(a * b),
                (l, r) => IndexExpr::Mul(Box::new(l), Box::new(r)),
            },
            IndexExpr::Div(l, r) => match (l.simplify(), r.simplify()) {
                (IndexExpr::Const(0), _) => IndexExpr::Const(0),
                (e, IndexExpr::Const(1)) => e,
                (IndexExpr::Const(a), IndexExpr::Const(b)) if b != 0 => IndexExpr::Const(a / b),
                (l, r) if l == r => IndexExpr::Const(1),
                (l, r) => IndexExpr::Div(Box::new(l), Box::new(r)),
            },
            IndexExpr::Min(l, r) => match (l.simplify(), r.simplify()) {
                (IndexExpr::Const(a), IndexExpr::Const(b)) => IndexExpr::Const(a.min(b)),
                (l, r) if l == r => l,
                (l, r) => IndexExpr::Min(Box::new(l), Box::new(r)),
            },
            IndexExpr::Max(l, r) => match (l.simplify(), r.simplify()) {
                (IndexExpr::Const(a), IndexExpr::Const(b)) => IndexExpr::Const(a.max(b)),
                (l, r) if l == r => l,
                (l, r) => IndexExpr::Max(Box::new(l), Box::new(r)),
            },
            _ => self.clone(),
        }
    }

    /// Try to evaluate to a constant value
    pub fn try_eval(&self) -> Option<i64> {
        match self {
            IndexExpr::Const(v) => Some(*v),
            IndexExpr::Add(l, r) => Some(l.try_eval()? + r.try_eval()?),
            IndexExpr::Sub(l, r) => Some(l.try_eval()? - r.try_eval()?),
            IndexExpr::Mul(l, r) => Some(l.try_eval()? * r.try_eval()?),
            IndexExpr::Div(l, r) => {
                let rv = r.try_eval()?;
                if rv != 0 {
                    Some(l.try_eval()? / rv)
                } else {
                    None
                }
            }
            IndexExpr::Min(l, r) => Some(l.try_eval()?.min(r.try_eval()?)),
            IndexExpr::Max(l, r) => Some(l.try_eval()?.max(r.try_eval()?)),
            IndexExpr::Var(_) => None,
        }
    }
}

impl fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexExpr::Var(name) => write!(f, "{}", name),
            IndexExpr::Const(v) => write!(f, "{}", v),
            IndexExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            IndexExpr::Sub(l, r) => write!(f, "({} - {})", l, r),
            IndexExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            IndexExpr::Div(l, r) => write!(f, "({} / {})", l, r),
            IndexExpr::Min(l, r) => write!(f, "min({}, {})", l, r),
            IndexExpr::Max(l, r) => write!(f, "max({}, {})", l, r),
        }
    }
}

/// Dimension constraints for dependent types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimConstraint {
    /// Equality: e1 == e2
    Eq(IndexExpr, IndexExpr),
    /// Less than: e1 < e2
    Lt(IndexExpr, IndexExpr),
    /// Less than or equal: e1 <= e2
    Lte(IndexExpr, IndexExpr),
    /// Greater than: e1 > e2
    Gt(IndexExpr, IndexExpr),
    /// Greater than or equal: e1 >= e2
    Gte(IndexExpr, IndexExpr),
    /// Conjunction: c1 ∧ c2
    And(Box<DimConstraint>, Box<DimConstraint>),
    /// Disjunction: c1 ∨ c2
    Or(Box<DimConstraint>, Box<DimConstraint>),
    /// Negation: ¬c
    Not(Box<DimConstraint>),
}

impl DimConstraint {
    pub fn eq(left: IndexExpr, right: IndexExpr) -> Self {
        DimConstraint::Eq(left, right)
    }

    pub fn lt(left: IndexExpr, right: IndexExpr) -> Self {
        DimConstraint::Lt(left, right)
    }

    pub fn lte(left: IndexExpr, right: IndexExpr) -> Self {
        DimConstraint::Lte(left, right)
    }

    pub fn gt(left: IndexExpr, right: IndexExpr) -> Self {
        DimConstraint::Gt(left, right)
    }

    pub fn gte(left: IndexExpr, right: IndexExpr) -> Self {
        DimConstraint::Gte(left, right)
    }

    pub fn and(left: DimConstraint, right: DimConstraint) -> Self {
        DimConstraint::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: DimConstraint, right: DimConstraint) -> Self {
        DimConstraint::Or(Box::new(left), Box::new(right))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(constraint: DimConstraint) -> Self {
        DimConstraint::Not(Box::new(constraint))
    }

    /// Get all index variables referenced in this constraint
    pub fn referenced_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_referenced_vars(&mut vars);
        vars
    }

    fn collect_referenced_vars(&self, vars: &mut HashSet<String>) {
        match self {
            DimConstraint::Eq(l, r)
            | DimConstraint::Lt(l, r)
            | DimConstraint::Lte(l, r)
            | DimConstraint::Gt(l, r)
            | DimConstraint::Gte(l, r) => {
                vars.extend(l.free_vars());
                vars.extend(r.free_vars());
            }
            DimConstraint::And(l, r) | DimConstraint::Or(l, r) => {
                l.collect_referenced_vars(vars);
                r.collect_referenced_vars(vars);
            }
            DimConstraint::Not(c) => c.collect_referenced_vars(vars),
        }
    }

    /// Simplify the constraint
    pub fn simplify(&self) -> DimConstraint {
        match self {
            DimConstraint::Eq(l, r) => DimConstraint::Eq(l.simplify(), r.simplify()),
            DimConstraint::Lt(l, r) => DimConstraint::Lt(l.simplify(), r.simplify()),
            DimConstraint::Lte(l, r) => DimConstraint::Lte(l.simplify(), r.simplify()),
            DimConstraint::Gt(l, r) => DimConstraint::Gt(l.simplify(), r.simplify()),
            DimConstraint::Gte(l, r) => DimConstraint::Gte(l.simplify(), r.simplify()),
            DimConstraint::And(l, r) => {
                DimConstraint::And(Box::new(l.simplify()), Box::new(r.simplify()))
            }
            DimConstraint::Or(l, r) => {
                DimConstraint::Or(Box::new(l.simplify()), Box::new(r.simplify()))
            }
            DimConstraint::Not(c) => DimConstraint::Not(Box::new(c.simplify())),
        }
    }
}

impl fmt::Display for DimConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimConstraint::Eq(l, r) => write!(f, "{} == {}", l, r),
            DimConstraint::Lt(l, r) => write!(f, "{} < {}", l, r),
            DimConstraint::Lte(l, r) => write!(f, "{} <= {}", l, r),
            DimConstraint::Gt(l, r) => write!(f, "{} > {}", l, r),
            DimConstraint::Gte(l, r) => write!(f, "{} >= {}", l, r),
            DimConstraint::And(l, r) => write!(f, "({} ∧ {})", l, r),
            DimConstraint::Or(l, r) => write!(f, "({} ∨ {})", l, r),
            DimConstraint::Not(c) => write!(f, "¬{}", c),
        }
    }
}

/// Dependent type: types that depend on runtime values.
///
/// Examples:
/// - `Vec<n, T>`: Vector of length n with elements of type T
/// - `Matrix<m, n, T>`: Matrix with dimensions m×n
/// - `(x: Int) -> Vec<x, Bool>`: Function returning a vector of length x
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependentType {
    /// Base parametric type (non-dependent)
    Base(ParametricType),
    /// Vector with dependent length: Vec<n, T>
    Vector {
        length: IndexExpr,
        element_type: Box<DependentType>,
    },
    /// Matrix with dependent dimensions: Matrix<rows, cols, T>
    Matrix {
        rows: IndexExpr,
        cols: IndexExpr,
        element_type: Box<DependentType>,
    },
    /// Tensor with dependent shape: Tensor<[d1, d2, ...], T>
    Tensor {
        shape: Vec<IndexExpr>,
        element_type: Box<DependentType>,
    },
    /// Dependent function type: (x: T1) -> T2(x)
    DependentFunction {
        param_name: String,
        param_type: Box<DependentType>,
        return_type: Box<DependentType>,
    },
    /// Refinement type: {x: T | P(x)}
    Refinement {
        var_name: String,
        base_type: Box<DependentType>,
        predicate: Term,
    },
    /// Constrained type: T where C
    Constrained {
        base_type: Box<DependentType>,
        constraints: Vec<DimConstraint>,
    },
}

impl DependentType {
    /// Create a base non-dependent type
    pub fn base(param_type: ParametricType) -> Self {
        DependentType::Base(param_type)
    }

    /// Create a dependent vector type
    pub fn vector(length: IndexExpr, element_type: impl Into<String>) -> Self {
        DependentType::Vector {
            length,
            element_type: Box::new(DependentType::Base(ParametricType::concrete(element_type))),
        }
    }

    /// Create a dependent matrix type
    pub fn matrix(rows: IndexExpr, cols: IndexExpr, element_type: impl Into<String>) -> Self {
        DependentType::Matrix {
            rows,
            cols,
            element_type: Box::new(DependentType::Base(ParametricType::concrete(element_type))),
        }
    }

    /// Create a dependent tensor type
    pub fn tensor(shape: Vec<IndexExpr>, element_type: impl Into<String>) -> Self {
        DependentType::Tensor {
            shape,
            element_type: Box::new(DependentType::Base(ParametricType::concrete(element_type))),
        }
    }

    /// Create a dependent function type
    pub fn dependent_function(
        param_name: impl Into<String>,
        param_type: DependentType,
        return_type: DependentType,
    ) -> Self {
        DependentType::DependentFunction {
            param_name: param_name.into(),
            param_type: Box::new(param_type),
            return_type: Box::new(return_type),
        }
    }

    /// Create a refinement type
    pub fn refinement(
        var_name: impl Into<String>,
        base_type: DependentType,
        predicate: Term,
    ) -> Self {
        DependentType::Refinement {
            var_name: var_name.into(),
            base_type: Box::new(base_type),
            predicate,
        }
    }

    /// Add constraints to a type
    pub fn with_constraints(self, constraints: Vec<DimConstraint>) -> Self {
        DependentType::Constrained {
            base_type: Box::new(self),
            constraints,
        }
    }

    /// Get all free index variables
    pub fn free_index_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_index_vars(&mut vars, &HashSet::new());
        vars
    }

    fn collect_free_index_vars(&self, vars: &mut HashSet<String>, bound: &HashSet<String>) {
        match self {
            DependentType::Base(_) => {}
            DependentType::Vector {
                length,
                element_type,
            } => {
                vars.extend(length.free_vars().difference(bound).cloned());
                element_type.collect_free_index_vars(vars, bound);
            }
            DependentType::Matrix {
                rows,
                cols,
                element_type,
            } => {
                vars.extend(rows.free_vars().difference(bound).cloned());
                vars.extend(cols.free_vars().difference(bound).cloned());
                element_type.collect_free_index_vars(vars, bound);
            }
            DependentType::Tensor {
                shape,
                element_type,
            } => {
                for dim in shape {
                    vars.extend(dim.free_vars().difference(bound).cloned());
                }
                element_type.collect_free_index_vars(vars, bound);
            }
            DependentType::DependentFunction {
                param_name,
                param_type,
                return_type,
            } => {
                param_type.collect_free_index_vars(vars, bound);
                let mut new_bound = bound.clone();
                new_bound.insert(param_name.clone());
                return_type.collect_free_index_vars(vars, &new_bound);
            }
            DependentType::Refinement {
                var_name: _,
                base_type,
                predicate: _,
            } => {
                base_type.collect_free_index_vars(vars, bound);
            }
            DependentType::Constrained {
                base_type,
                constraints,
            } => {
                base_type.collect_free_index_vars(vars, bound);
                for constraint in constraints {
                    vars.extend(constraint.referenced_vars().difference(bound).cloned());
                }
            }
        }
    }

    /// Check if this type is well-formed (no unbound index variables)
    pub fn is_well_formed(&self) -> bool {
        self.free_index_vars().is_empty()
    }
}

impl fmt::Display for DependentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependentType::Base(t) => write!(f, "{}", t),
            DependentType::Vector {
                length,
                element_type,
            } => write!(f, "Vec<{}, {}>", length, element_type),
            DependentType::Matrix {
                rows,
                cols,
                element_type,
            } => write!(f, "Matrix<{}, {}, {}>", rows, cols, element_type),
            DependentType::Tensor {
                shape,
                element_type,
            } => {
                write!(f, "Tensor<[")?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", dim)?;
                }
                write!(f, "], {}>", element_type)
            }
            DependentType::DependentFunction {
                param_name,
                param_type,
                return_type,
            } => write!(f, "({}: {}) -> {}", param_name, param_type, return_type),
            DependentType::Refinement {
                var_name,
                base_type,
                predicate,
            } => write!(f, "{{{}:{} | {}}}", var_name, base_type, predicate),
            DependentType::Constrained {
                base_type,
                constraints,
            } => {
                write!(f, "{} where ", base_type)?;
                for (i, c) in constraints.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", c)?;
                }
                Ok(())
            }
        }
    }
}

/// Type checking context for dependent types.
#[derive(Clone, Debug, Default)]
pub struct DependentTypeContext {
    /// Index variable bindings
    index_bindings: HashMap<String, i64>,
    /// Dimension constraints
    constraints: Vec<DimConstraint>,
}

impl DependentTypeContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind an index variable to a value
    pub fn bind_index(&mut self, name: impl Into<String>, value: i64) {
        self.index_bindings.insert(name.into(), value);
    }

    /// Add a dimension constraint
    pub fn add_constraint(&mut self, constraint: DimConstraint) {
        self.constraints.push(constraint);
    }

    /// Check if constraints are satisfiable (simplified check)
    pub fn is_satisfiable(&self) -> bool {
        // For now, just check if we can evaluate all constraints with current bindings
        for constraint in &self.constraints {
            if !self.check_constraint(constraint) {
                return false;
            }
        }
        true
    }

    fn check_constraint(&self, constraint: &DimConstraint) -> bool {
        match constraint {
            DimConstraint::Eq(l, r) => {
                let lv = self.eval_index(l);
                let rv = self.eval_index(r);
                match (lv, rv) {
                    (Some(a), Some(b)) => a == b,
                    _ => true, // Unknown, assume satisfiable
                }
            }
            DimConstraint::Lt(l, r) => {
                let lv = self.eval_index(l);
                let rv = self.eval_index(r);
                match (lv, rv) {
                    (Some(a), Some(b)) => a < b,
                    _ => true,
                }
            }
            DimConstraint::Lte(l, r) => {
                let lv = self.eval_index(l);
                let rv = self.eval_index(r);
                match (lv, rv) {
                    (Some(a), Some(b)) => a <= b,
                    _ => true,
                }
            }
            DimConstraint::Gt(l, r) => {
                let lv = self.eval_index(l);
                let rv = self.eval_index(r);
                match (lv, rv) {
                    (Some(a), Some(b)) => a > b,
                    _ => true,
                }
            }
            DimConstraint::Gte(l, r) => {
                let lv = self.eval_index(l);
                let rv = self.eval_index(r);
                match (lv, rv) {
                    (Some(a), Some(b)) => a >= b,
                    _ => true,
                }
            }
            DimConstraint::And(l, r) => self.check_constraint(l) && self.check_constraint(r),
            DimConstraint::Or(l, r) => self.check_constraint(l) || self.check_constraint(r),
            DimConstraint::Not(c) => !self.check_constraint(c),
        }
    }

    fn eval_index(&self, expr: &IndexExpr) -> Option<i64> {
        match expr {
            IndexExpr::Var(name) => self.index_bindings.get(name).copied(),
            IndexExpr::Const(v) => Some(*v),
            IndexExpr::Add(l, r) => Some(self.eval_index(l)? + self.eval_index(r)?),
            IndexExpr::Sub(l, r) => Some(self.eval_index(l)? - self.eval_index(r)?),
            IndexExpr::Mul(l, r) => Some(self.eval_index(l)? * self.eval_index(r)?),
            IndexExpr::Div(l, r) => {
                let rv = self.eval_index(r)?;
                if rv != 0 {
                    Some(self.eval_index(l)? / rv)
                } else {
                    None
                }
            }
            IndexExpr::Min(l, r) => Some(self.eval_index(l)?.min(self.eval_index(r)?)),
            IndexExpr::Max(l, r) => Some(self.eval_index(l)?.max(self.eval_index(r)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_expr_basics() {
        let n = IndexExpr::var("n");
        let m = IndexExpr::var("m");
        let c = IndexExpr::constant(10);

        assert_eq!(n.to_string(), "n");
        assert_eq!(c.to_string(), "10");
        assert_eq!(IndexExpr::add(n.clone(), m.clone()).to_string(), "(n + m)");
    }

    #[test]
    fn test_index_expr_simplification() {
        let n = IndexExpr::var("n");
        let zero = IndexExpr::constant(0);
        let one = IndexExpr::constant(1);

        // n + 0 = n
        let expr = IndexExpr::add(n.clone(), zero.clone());
        assert_eq!(expr.simplify(), n);

        // n * 1 = n
        let expr = IndexExpr::mul(n.clone(), one.clone());
        assert_eq!(expr.simplify(), n);

        // n * 0 = 0
        let expr = IndexExpr::mul(n.clone(), zero.clone());
        assert_eq!(expr.simplify(), zero);

        // 5 + 3 = 8
        let expr = IndexExpr::add(IndexExpr::constant(5), IndexExpr::constant(3));
        assert_eq!(expr.simplify(), IndexExpr::constant(8));
    }

    #[test]
    fn test_index_expr_eval() {
        let expr = IndexExpr::add(IndexExpr::constant(5), IndexExpr::constant(3));
        assert_eq!(expr.try_eval(), Some(8));

        let expr = IndexExpr::mul(IndexExpr::constant(4), IndexExpr::constant(7));
        assert_eq!(expr.try_eval(), Some(28));

        let expr = IndexExpr::add(IndexExpr::var("n"), IndexExpr::constant(5));
        assert_eq!(expr.try_eval(), None);
    }

    #[test]
    fn test_dependent_vector_type() {
        let n = IndexExpr::var("n");
        let vec_type = DependentType::vector(n.clone(), "Int");

        assert_eq!(vec_type.to_string(), "Vec<n, Int>");
        assert_eq!(vec_type.free_index_vars(), {
            let mut s = HashSet::new();
            s.insert("n".to_string());
            s
        });
    }

    #[test]
    fn test_dependent_matrix_type() {
        let m = IndexExpr::var("m");
        let n = IndexExpr::var("n");
        let matrix_type = DependentType::matrix(m, n, "Float");

        assert_eq!(matrix_type.to_string(), "Matrix<m, n, Float>");
    }

    #[test]
    fn test_dependent_tensor_type() {
        let d1 = IndexExpr::var("d1");
        let d2 = IndexExpr::var("d2");
        let d3 = IndexExpr::constant(10);

        let tensor_type = DependentType::tensor(vec![d1, d2, d3], "Float");
        assert_eq!(tensor_type.to_string(), "Tensor<[d1, d2, 10], Float>");
    }

    #[test]
    fn test_dependent_function_type() {
        let n_param = DependentType::base(ParametricType::concrete("Int"));
        let n_var = IndexExpr::var("n");
        let return_type = DependentType::vector(n_var, "Bool");

        let func_type = DependentType::dependent_function("n", n_param, return_type);
        assert_eq!(func_type.to_string(), "(n: Int) -> Vec<n, Bool>");
    }

    #[test]
    fn test_dimension_constraints() {
        let n = IndexExpr::var("n");
        let m = IndexExpr::var("m");

        let c1 = DimConstraint::lt(n.clone(), IndexExpr::constant(100));
        let c2 = DimConstraint::gte(n.clone(), IndexExpr::constant(0));
        let c3 = DimConstraint::eq(n.clone(), m.clone());

        assert_eq!(c1.to_string(), "n < 100");
        assert_eq!(c2.to_string(), "n >= 0");
        assert_eq!(c3.to_string(), "n == m");

        let combined = DimConstraint::and(c1, c2);
        assert_eq!(combined.to_string(), "(n < 100 ∧ n >= 0)");
    }

    #[test]
    fn test_constrained_type() {
        let n = IndexExpr::var("n");
        let vec_type = DependentType::vector(n.clone(), "Int");

        let constraint = DimConstraint::lte(n.clone(), IndexExpr::constant(100));
        let constrained = vec_type.with_constraints(vec![constraint]);

        assert_eq!(constrained.to_string(), "Vec<n, Int> where n <= 100");
    }

    #[test]
    fn test_type_context_satisfiability() {
        let mut ctx = DependentTypeContext::new();
        ctx.bind_index("n", 50);

        let constraint = DimConstraint::lte(IndexExpr::var("n"), IndexExpr::constant(100));
        ctx.add_constraint(constraint);

        assert!(ctx.is_satisfiable());

        let bad_constraint = DimConstraint::gt(IndexExpr::var("n"), IndexExpr::constant(100));
        ctx.add_constraint(bad_constraint);

        assert!(!ctx.is_satisfiable());
    }

    #[test]
    fn test_refinement_type() {
        let base = DependentType::base(ParametricType::concrete("Int"));
        let predicate = Term::var("x"); // Simplified predicate

        let refined = DependentType::refinement("x", base, predicate);
        assert!(refined.to_string().contains("{x:Int |"));
    }

    #[test]
    fn test_free_index_vars_in_complex_type() {
        // (n: Int) -> Matrix<n, n, Float>
        let n_param = DependentType::base(ParametricType::concrete("Int"));
        let n_var = IndexExpr::var("n");
        let return_type = DependentType::matrix(n_var.clone(), n_var, "Float");

        let func_type = DependentType::dependent_function("n", n_param, return_type);

        // 'n' should be bound in the function, so no free variables
        assert!(func_type.is_well_formed());
    }

    #[test]
    fn test_index_substitution() {
        let n = IndexExpr::var("n");
        let m = IndexExpr::var("m");
        let expr = IndexExpr::add(n.clone(), m.clone());

        let mut subst = HashMap::new();
        subst.insert("n".to_string(), IndexExpr::constant(10));

        let result = expr.substitute(&subst);
        assert_eq!(result.to_string(), "(10 + m)");
    }
}
