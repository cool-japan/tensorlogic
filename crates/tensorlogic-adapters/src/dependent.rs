//! Dependent types for expressing value-dependent type constraints.
//!
//! Dependent types allow types to depend on values, enabling precise specification
//! of tensor dimensions, vector lengths, and other parameterized types.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{DependentType, DimExpr, DependentTypeContext};
//!
//! // Create a vector type with dependent length
//! let vec_n = DependentType::vector("T", DimExpr::var("n"));
//!
//! // Create a matrix type with dependent dimensions
//! let matrix = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"));
//!
//! // Evaluate dimensions in context
//! let mut ctx = DependentTypeContext::new();
//! ctx.set_dim("n", 10);
//! ctx.set_dim("m", 5);
//!
//! assert_eq!(vec_n.eval_shape(&ctx), Some(vec![10]));
//! assert_eq!(matrix.eval_shape(&ctx), Some(vec![5, 10]));
//! ```

use std::collections::HashMap;
use std::fmt;

/// A dimension expression that can be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    /// A concrete dimension value
    Const(usize),
    /// A dimension variable
    Var(String),
    /// Addition of dimensions
    Add(Box<DimExpr>, Box<DimExpr>),
    /// Subtraction of dimensions
    Sub(Box<DimExpr>, Box<DimExpr>),
    /// Multiplication of dimensions
    Mul(Box<DimExpr>, Box<DimExpr>),
    /// Division of dimensions (integer division)
    Div(Box<DimExpr>, Box<DimExpr>),
    /// Maximum of two dimensions
    Max(Box<DimExpr>, Box<DimExpr>),
    /// Minimum of two dimensions
    Min(Box<DimExpr>, Box<DimExpr>),
    /// Ceiling division (useful for padding/strides)
    CeilDiv(Box<DimExpr>, Box<DimExpr>),
}

impl DimExpr {
    /// Create a constant dimension.
    pub fn constant(value: usize) -> Self {
        DimExpr::Const(value)
    }

    /// Create a dimension variable.
    pub fn var(name: impl Into<String>) -> Self {
        DimExpr::Var(name.into())
    }

    /// Add two dimension expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: DimExpr) -> Self {
        DimExpr::Add(Box::new(self), Box::new(other))
    }

    /// Subtract a dimension expression from this one.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: DimExpr) -> Self {
        DimExpr::Sub(Box::new(self), Box::new(other))
    }

    /// Multiply two dimension expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: DimExpr) -> Self {
        DimExpr::Mul(Box::new(self), Box::new(other))
    }

    /// Divide this dimension expression by another.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: DimExpr) -> Self {
        DimExpr::Div(Box::new(self), Box::new(other))
    }

    /// Take the maximum of two dimension expressions.
    pub fn max(self, other: DimExpr) -> Self {
        DimExpr::Max(Box::new(self), Box::new(other))
    }

    /// Take the minimum of two dimension expressions.
    pub fn min(self, other: DimExpr) -> Self {
        DimExpr::Min(Box::new(self), Box::new(other))
    }

    /// Ceiling division.
    pub fn ceil_div(self, other: DimExpr) -> Self {
        DimExpr::CeilDiv(Box::new(self), Box::new(other))
    }

    /// Evaluate the dimension expression in a context.
    pub fn eval(&self, ctx: &DependentTypeContext) -> Option<usize> {
        match self {
            DimExpr::Const(n) => Some(*n),
            DimExpr::Var(name) => ctx.get_dim(name).copied(),
            DimExpr::Add(a, b) => Some(a.eval(ctx)? + b.eval(ctx)?),
            DimExpr::Sub(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                av.checked_sub(bv)
            }
            DimExpr::Mul(a, b) => Some(a.eval(ctx)? * b.eval(ctx)?),
            DimExpr::Div(a, b) => {
                let bv = b.eval(ctx)?;
                if bv == 0 {
                    return None;
                }
                Some(a.eval(ctx)? / bv)
            }
            DimExpr::Max(a, b) => Some(a.eval(ctx)?.max(b.eval(ctx)?)),
            DimExpr::Min(a, b) => Some(a.eval(ctx)?.min(b.eval(ctx)?)),
            DimExpr::CeilDiv(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                if bv == 0 {
                    return None;
                }
                Some(av.div_ceil(bv))
            }
        }
    }

    /// Get all free variables in this expression.
    pub fn free_variables(&self) -> Vec<String> {
        match self {
            DimExpr::Const(_) => vec![],
            DimExpr::Var(name) => vec![name.clone()],
            DimExpr::Add(a, b)
            | DimExpr::Sub(a, b)
            | DimExpr::Mul(a, b)
            | DimExpr::Div(a, b)
            | DimExpr::Max(a, b)
            | DimExpr::Min(a, b)
            | DimExpr::CeilDiv(a, b) => {
                let mut vars = a.free_variables();
                vars.extend(b.free_variables());
                vars.sort();
                vars.dedup();
                vars
            }
        }
    }

    /// Substitute a variable with an expression.
    pub fn substitute(&self, var: &str, expr: &DimExpr) -> DimExpr {
        match self {
            DimExpr::Const(n) => DimExpr::Const(*n),
            DimExpr::Var(name) => {
                if name == var {
                    expr.clone()
                } else {
                    DimExpr::Var(name.clone())
                }
            }
            DimExpr::Add(a, b) => DimExpr::Add(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::Sub(a, b) => DimExpr::Sub(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::Mul(a, b) => DimExpr::Mul(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::Div(a, b) => DimExpr::Div(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::Max(a, b) => DimExpr::Max(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::Min(a, b) => DimExpr::Min(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
            DimExpr::CeilDiv(a, b) => DimExpr::CeilDiv(
                Box::new(a.substitute(var, expr)),
                Box::new(b.substitute(var, expr)),
            ),
        }
    }

    /// Simplify the expression.
    pub fn simplify(&self) -> DimExpr {
        match self {
            DimExpr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) => DimExpr::Const(x + y),
                    (DimExpr::Const(0), _) => b,
                    (_, DimExpr::Const(0)) => a,
                    _ => DimExpr::Add(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) => DimExpr::Const(x.saturating_sub(*y)),
                    (_, DimExpr::Const(0)) => a,
                    _ => DimExpr::Sub(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) => DimExpr::Const(x * y),
                    (DimExpr::Const(0), _) | (_, DimExpr::Const(0)) => DimExpr::Const(0),
                    (DimExpr::Const(1), _) => b,
                    (_, DimExpr::Const(1)) => a,
                    _ => DimExpr::Mul(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) if *y != 0 => DimExpr::Const(x / y),
                    (DimExpr::Const(0), _) => DimExpr::Const(0),
                    (_, DimExpr::Const(1)) => a,
                    _ => DimExpr::Div(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::Max(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) => DimExpr::Const((*x).max(*y)),
                    _ => DimExpr::Max(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::Min(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) => DimExpr::Const((*x).min(*y)),
                    _ => DimExpr::Min(Box::new(a), Box::new(b)),
                }
            }
            DimExpr::CeilDiv(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (DimExpr::Const(x), DimExpr::Const(y)) if *y != 0 => {
                        DimExpr::Const(x.div_ceil(*y))
                    }
                    _ => DimExpr::CeilDiv(Box::new(a), Box::new(b)),
                }
            }
            other => other.clone(),
        }
    }

    /// Check if two dimension expressions are equal.
    pub fn is_equal(&self, other: &DimExpr, ctx: &DependentTypeContext) -> bool {
        // Try symbolic equality first
        if self == other {
            return true;
        }

        // Try numeric equality
        match (self.eval(ctx), other.eval(ctx)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Const(n) => write!(f, "{}", n),
            DimExpr::Var(name) => write!(f, "{}", name),
            DimExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            DimExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            DimExpr::Mul(a, b) => write!(f, "({} * {})", a, b),
            DimExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            DimExpr::Max(a, b) => write!(f, "max({}, {})", a, b),
            DimExpr::Min(a, b) => write!(f, "min({}, {})", a, b),
            DimExpr::CeilDiv(a, b) => write!(f, "ceil({} / {})", a, b),
        }
    }
}

/// A dependent type with dimension parameters.
#[derive(Debug, Clone)]
pub struct DependentType {
    /// Base type name
    pub base_type: String,
    /// Type parameters (other types)
    pub type_params: Vec<String>,
    /// Dimension parameters
    pub dim_params: Vec<DimExpr>,
    /// Optional name for the dependent type
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Constraints between dimensions
    pub constraints: Vec<DimConstraint>,
}

/// A constraint between dimension expressions.
#[derive(Debug, Clone)]
pub struct DimConstraint {
    /// Left-hand side expression
    pub lhs: DimExpr,
    /// Constraint relation
    pub relation: DimRelation,
    /// Right-hand side expression
    pub rhs: DimExpr,
    /// Error message if constraint is violated
    pub message: Option<String>,
}

/// Relation for dimension constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimRelation {
    /// Equal
    Equal,
    /// Not equal
    NotEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Divisible by
    DivisibleBy,
}

impl DependentType {
    /// Create a new dependent type with a base type.
    pub fn new(base_type: impl Into<String>) -> Self {
        DependentType {
            base_type: base_type.into(),
            type_params: Vec::new(),
            dim_params: Vec::new(),
            name: None,
            description: None,
            constraints: Vec::new(),
        }
    }

    /// Create a scalar type.
    pub fn scalar(element_type: impl Into<String>) -> Self {
        DependentType::new(element_type)
    }

    /// Create a vector type with a dimension.
    pub fn vector(element_type: impl Into<String>, length: DimExpr) -> Self {
        DependentType {
            base_type: "Vector".to_string(),
            type_params: vec![element_type.into()],
            dim_params: vec![length],
            name: None,
            description: None,
            constraints: Vec::new(),
        }
    }

    /// Create a matrix type with dimensions.
    pub fn matrix(element_type: impl Into<String>, rows: DimExpr, cols: DimExpr) -> Self {
        DependentType {
            base_type: "Matrix".to_string(),
            type_params: vec![element_type.into()],
            dim_params: vec![rows, cols],
            name: None,
            description: None,
            constraints: Vec::new(),
        }
    }

    /// Create a tensor type with arbitrary dimensions.
    pub fn tensor(element_type: impl Into<String>, dims: Vec<DimExpr>) -> Self {
        DependentType {
            base_type: "Tensor".to_string(),
            type_params: vec![element_type.into()],
            dim_params: dims,
            name: None,
            description: None,
            constraints: Vec::new(),
        }
    }

    /// Set the name of this type.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a type parameter.
    pub fn with_type_param(mut self, param: impl Into<String>) -> Self {
        self.type_params.push(param.into());
        self
    }

    /// Add a dimension parameter.
    pub fn with_dim_param(mut self, dim: DimExpr) -> Self {
        self.dim_params.push(dim);
        self
    }

    /// Add a dimension constraint.
    pub fn with_constraint(mut self, constraint: DimConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Get the effective name.
    pub fn type_name(&self) -> String {
        if let Some(name) = &self.name {
            return name.clone();
        }

        if self.dim_params.is_empty() && self.type_params.is_empty() {
            return self.base_type.clone();
        }

        let mut result = self.base_type.clone();
        if !self.type_params.is_empty() || !self.dim_params.is_empty() {
            result.push('<');

            let mut parts = Vec::new();
            for tp in &self.type_params {
                parts.push(tp.clone());
            }
            for dp in &self.dim_params {
                parts.push(format!("{}", dp));
            }

            result.push_str(&parts.join(", "));
            result.push('>');
        }
        result
    }

    /// Evaluate the shape of this type in a context.
    pub fn eval_shape(&self, ctx: &DependentTypeContext) -> Option<Vec<usize>> {
        self.dim_params.iter().map(|d| d.eval(ctx)).collect()
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dim_params.len()
    }

    /// Get all free dimension variables.
    pub fn free_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for dim in &self.dim_params {
            vars.extend(dim.free_variables());
        }
        for constraint in &self.constraints {
            vars.extend(constraint.lhs.free_variables());
            vars.extend(constraint.rhs.free_variables());
        }
        vars.sort();
        vars.dedup();
        vars
    }

    /// Check if constraints are satisfied in a context.
    pub fn check_constraints(&self, ctx: &DependentTypeContext) -> Result<(), String> {
        for constraint in &self.constraints {
            if !constraint.check(ctx) {
                let msg = constraint.message.clone().unwrap_or_else(|| {
                    format!(
                        "Constraint violated: {} {:?} {}",
                        constraint.lhs, constraint.relation, constraint.rhs
                    )
                });
                return Err(msg);
            }
        }
        Ok(())
    }

    /// Check if this type is compatible with another for assignment.
    pub fn is_compatible_with(&self, other: &DependentType, ctx: &DependentTypeContext) -> bool {
        // Base types must match
        if self.base_type != other.base_type {
            return false;
        }

        // Type parameters must match
        if self.type_params != other.type_params {
            return false;
        }

        // Dimension parameters must be equal
        if self.dim_params.len() != other.dim_params.len() {
            return false;
        }

        for (a, b) in self.dim_params.iter().zip(&other.dim_params) {
            if !a.is_equal(b, ctx) {
                return false;
            }
        }

        true
    }
}

impl DimConstraint {
    /// Create a new dimension constraint.
    pub fn new(lhs: DimExpr, relation: DimRelation, rhs: DimExpr) -> Self {
        DimConstraint {
            lhs,
            relation,
            rhs,
            message: None,
        }
    }

    /// Set the error message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Check if the constraint is satisfied.
    pub fn check(&self, ctx: &DependentTypeContext) -> bool {
        let lhs_val = match self.lhs.eval(ctx) {
            Some(v) => v,
            None => return false,
        };
        let rhs_val = match self.rhs.eval(ctx) {
            Some(v) => v,
            None => return false,
        };

        match self.relation {
            DimRelation::Equal => lhs_val == rhs_val,
            DimRelation::NotEqual => lhs_val != rhs_val,
            DimRelation::LessThan => lhs_val < rhs_val,
            DimRelation::LessThanOrEqual => lhs_val <= rhs_val,
            DimRelation::GreaterThan => lhs_val > rhs_val,
            DimRelation::GreaterThanOrEqual => lhs_val >= rhs_val,
            DimRelation::DivisibleBy => rhs_val != 0 && lhs_val % rhs_val == 0,
        }
    }
}

/// Context for evaluating dependent types.
#[derive(Debug, Clone, Default)]
pub struct DependentTypeContext {
    /// Dimension variable values
    dims: HashMap<String, usize>,
    /// Type definitions
    types: HashMap<String, DependentType>,
}

impl DependentTypeContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        DependentTypeContext {
            dims: HashMap::new(),
            types: HashMap::new(),
        }
    }

    /// Set a dimension variable.
    pub fn set_dim(&mut self, name: impl Into<String>, value: usize) {
        self.dims.insert(name.into(), value);
    }

    /// Get a dimension variable.
    pub fn get_dim(&self, name: &str) -> Option<&usize> {
        self.dims.get(name)
    }

    /// Register a type definition.
    pub fn set_type(&mut self, name: impl Into<String>, ty: DependentType) {
        self.types.insert(name.into(), ty);
    }

    /// Get a type definition.
    pub fn get_type(&self, name: &str) -> Option<&DependentType> {
        self.types.get(name)
    }

    /// Check if a dimension variable exists.
    pub fn has_dim(&self, name: &str) -> bool {
        self.dims.contains_key(name)
    }

    /// Get all dimension variable names.
    pub fn dim_names(&self) -> Vec<&str> {
        self.dims.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all dimension bindings.
    pub fn clear_dims(&mut self) {
        self.dims.clear();
    }

    /// Unify two dimension expressions if possible.
    ///
    /// Returns true if unification succeeds and updates the context.
    pub fn unify(&mut self, a: &DimExpr, b: &DimExpr) -> bool {
        match (a, b) {
            (DimExpr::Var(va), DimExpr::Var(vb)) if va == vb => true,
            (DimExpr::Var(va), expr) | (expr, DimExpr::Var(va)) => {
                if let Some(&existing) = self.dims.get(va) {
                    if let Some(val) = expr.eval(self) {
                        existing == val
                    } else {
                        false
                    }
                } else if let Some(val) = expr.eval(self) {
                    self.dims.insert(va.clone(), val);
                    true
                } else {
                    false
                }
            }
            (DimExpr::Const(ca), DimExpr::Const(cb)) => ca == cb,
            _ => {
                // Try to evaluate both
                match (a.eval(self), b.eval(self)) {
                    (Some(va), Some(vb)) => va == vb,
                    _ => false,
                }
            }
        }
    }
}

/// Registry for managing dependent types.
#[derive(Debug, Clone, Default)]
pub struct DependentTypeRegistry {
    /// Named dependent types
    types: HashMap<String, DependentType>,
}

impl DependentTypeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        DependentTypeRegistry {
            types: HashMap::new(),
        }
    }

    /// Register a dependent type.
    pub fn register(&mut self, ty: DependentType) {
        let name = ty.type_name();
        self.types.insert(name, ty);
    }

    /// Get a type by name.
    pub fn get(&self, name: &str) -> Option<&DependentType> {
        self.types.get(name)
    }

    /// Check if a type exists.
    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Get all type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

/// Common dependent type patterns.
pub mod patterns {
    use super::*;

    /// Square matrix type.
    pub fn square_matrix(element_type: impl Into<String>, size: DimExpr) -> DependentType {
        DependentType::matrix(element_type, size.clone(), size).with_name("SquareMatrix")
    }

    /// Identity matrix type (square with ones on diagonal).
    pub fn identity_matrix(size: DimExpr) -> DependentType {
        DependentType::matrix("Float", size.clone(), size).with_name("IdentityMatrix")
    }

    /// Batch of vectors.
    pub fn batch_vector(
        element_type: impl Into<String>,
        batch: DimExpr,
        length: DimExpr,
    ) -> DependentType {
        DependentType::tensor(element_type, vec![batch, length]).with_name("BatchVector")
    }

    /// Batch of matrices.
    pub fn batch_matrix(
        element_type: impl Into<String>,
        batch: DimExpr,
        rows: DimExpr,
        cols: DimExpr,
    ) -> DependentType {
        DependentType::tensor(element_type, vec![batch, rows, cols]).with_name("BatchMatrix")
    }

    /// 4D image tensor (batch, channels, height, width).
    pub fn image_tensor(
        batch: DimExpr,
        channels: DimExpr,
        height: DimExpr,
        width: DimExpr,
    ) -> DependentType {
        DependentType::tensor("Float", vec![batch, channels, height, width])
            .with_name("ImageTensor")
    }

    /// Attention tensor (batch, heads, seq_len, head_dim).
    pub fn attention_tensor(
        batch: DimExpr,
        heads: DimExpr,
        seq_len: DimExpr,
        head_dim: DimExpr,
    ) -> DependentType {
        DependentType::tensor("Float", vec![batch, heads, seq_len, head_dim])
            .with_name("AttentionTensor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_expr_const() {
        let expr = DimExpr::Const(42);
        let ctx = DependentTypeContext::new();
        assert_eq!(expr.eval(&ctx), Some(42));
    }

    #[test]
    fn test_dim_expr_var() {
        let expr = DimExpr::Var("n".to_string());
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("n", 10);
        assert_eq!(expr.eval(&ctx), Some(10));
    }

    #[test]
    fn test_dim_expr_arithmetic() {
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("n", 10);
        ctx.set_dim("m", 3);

        let add = DimExpr::var("n").add(DimExpr::var("m"));
        assert_eq!(add.eval(&ctx), Some(13));

        let mul = DimExpr::var("n").mul(DimExpr::var("m"));
        assert_eq!(mul.eval(&ctx), Some(30));

        let div = DimExpr::var("n").div(DimExpr::var("m"));
        assert_eq!(div.eval(&ctx), Some(3));
    }

    #[test]
    fn test_dim_expr_max_min() {
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("a", 5);
        ctx.set_dim("b", 10);

        let max = DimExpr::var("a").max(DimExpr::var("b"));
        assert_eq!(max.eval(&ctx), Some(10));

        let min = DimExpr::var("a").min(DimExpr::var("b"));
        assert_eq!(min.eval(&ctx), Some(5));
    }

    #[test]
    fn test_dim_expr_simplify() {
        let expr = DimExpr::constant(5).add(DimExpr::constant(3));
        let simplified = expr.simplify();
        assert_eq!(simplified, DimExpr::Const(8));

        let expr = DimExpr::var("x").add(DimExpr::constant(0));
        let simplified = expr.simplify();
        assert_eq!(simplified, DimExpr::Var("x".to_string()));
    }

    #[test]
    fn test_vector_type() {
        let vec_ty = DependentType::vector("Float", DimExpr::var("n"));
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("n", 100);

        assert_eq!(vec_ty.eval_shape(&ctx), Some(vec![100]));
        assert_eq!(vec_ty.rank(), 1);
    }

    #[test]
    fn test_matrix_type() {
        let mat_ty = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"));
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("m", 10);
        ctx.set_dim("n", 20);

        assert_eq!(mat_ty.eval_shape(&ctx), Some(vec![10, 20]));
        assert_eq!(mat_ty.rank(), 2);
    }

    #[test]
    fn test_dim_constraint() {
        let constraint = DimConstraint::new(
            DimExpr::var("n"),
            DimRelation::GreaterThan,
            DimExpr::constant(0),
        );

        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("n", 10);
        assert!(constraint.check(&ctx));

        ctx.set_dim("n", 0);
        assert!(!constraint.check(&ctx));
    }

    #[test]
    fn test_type_with_constraints() {
        let ty = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"))
            .with_constraint(
                DimConstraint::new(DimExpr::var("m"), DimRelation::Equal, DimExpr::var("n"))
                    .with_message("Matrix must be square"),
            );

        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("m", 10);
        ctx.set_dim("n", 10);
        assert!(ty.check_constraints(&ctx).is_ok());

        ctx.set_dim("n", 20);
        assert!(ty.check_constraints(&ctx).is_err());
    }

    #[test]
    fn test_type_compatibility() {
        let ty1 = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"));
        let ty2 = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"));

        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("m", 10);
        ctx.set_dim("n", 20);

        assert!(ty1.is_compatible_with(&ty2, &ctx));

        let ty3 = DependentType::matrix("Int", DimExpr::var("m"), DimExpr::var("n"));
        assert!(!ty1.is_compatible_with(&ty3, &ctx));
    }

    #[test]
    fn test_free_variables() {
        let expr = DimExpr::var("n")
            .add(DimExpr::var("m"))
            .mul(DimExpr::var("k"));
        let vars = expr.free_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&"k".to_string()));
        assert!(vars.contains(&"m".to_string()));
        assert!(vars.contains(&"n".to_string()));
    }

    #[test]
    fn test_substitute() {
        let expr = DimExpr::var("n").add(DimExpr::constant(5));
        let substituted = expr.substitute("n", &DimExpr::constant(10));

        let ctx = DependentTypeContext::new();
        assert_eq!(substituted.eval(&ctx), Some(15));
    }

    #[test]
    fn test_ceil_div() {
        let expr = DimExpr::constant(10).ceil_div(DimExpr::constant(3));
        let ctx = DependentTypeContext::new();
        assert_eq!(expr.eval(&ctx), Some(4)); // ceil(10/3) = 4
    }

    #[test]
    fn test_context_unify() {
        let mut ctx = DependentTypeContext::new();

        // Unify variable with constant
        let success = ctx.unify(&DimExpr::var("n"), &DimExpr::constant(10));
        assert!(success);
        assert_eq!(ctx.get_dim("n"), Some(&10));

        // Unify with existing binding
        let success = ctx.unify(&DimExpr::var("n"), &DimExpr::constant(10));
        assert!(success);

        // Fail to unify incompatible
        let success = ctx.unify(&DimExpr::var("n"), &DimExpr::constant(20));
        assert!(!success);
    }

    #[test]
    fn test_patterns() {
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("n", 64);
        ctx.set_dim("batch", 32);
        ctx.set_dim("heads", 8);
        ctx.set_dim("seq_len", 512);
        ctx.set_dim("head_dim", 64);

        let sq = patterns::square_matrix("Float", DimExpr::var("n"));
        assert_eq!(sq.eval_shape(&ctx), Some(vec![64, 64]));

        let attn = patterns::attention_tensor(
            DimExpr::var("batch"),
            DimExpr::var("heads"),
            DimExpr::var("seq_len"),
            DimExpr::var("head_dim"),
        );
        assert_eq!(attn.eval_shape(&ctx), Some(vec![32, 8, 512, 64]));
    }

    #[test]
    fn test_registry() {
        let mut registry = DependentTypeRegistry::new();

        registry
            .register(DependentType::vector("Float", DimExpr::var("n")).with_name("FloatVector"));

        assert!(registry.contains("FloatVector"));
        assert_eq!(registry.len(), 1);

        let ty = registry.get("FloatVector").unwrap();
        assert_eq!(ty.base_type, "Vector");
    }

    #[test]
    fn test_dim_display() {
        let expr = DimExpr::var("n").add(DimExpr::var("m"));
        assert_eq!(format!("{}", expr), "(n + m)");

        let expr = DimExpr::var("a").mul(DimExpr::constant(2));
        assert_eq!(format!("{}", expr), "(a * 2)");
    }

    #[test]
    fn test_type_name() {
        let ty = DependentType::matrix("Float", DimExpr::var("m"), DimExpr::var("n"));
        assert_eq!(ty.type_name(), "Matrix<Float, m, n>");

        let ty = ty.with_name("MyMatrix");
        assert_eq!(ty.type_name(), "MyMatrix");
    }
}
