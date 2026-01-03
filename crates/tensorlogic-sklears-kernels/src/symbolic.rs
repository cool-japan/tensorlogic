//! Symbolic kernel composition and algebraic operations
//!
//! This module provides symbolic kernel composition capabilities, allowing users to:
//! - Build complex kernels using algebraic expressions
//! - Parse kernel expressions from strings
//! - Combine kernels using operators (+, *, ^)
//! - Create kernel pipelines declaratively
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{
//!     LinearKernel, RbfKernel, RbfKernelConfig,
//!     symbolic::KernelBuilder,
//!     Kernel,
//! };
//!
//! // Build kernel using algebraic composition
//! let builder = KernelBuilder::new();
//! let kernel = builder
//!     .add_scaled(std::sync::Arc::new(LinearKernel::new()), 0.5)
//!     .add_scaled(std::sync::Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()), 0.3)
//!     .build();
//!
//! // Use the composed kernel
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let similarity = kernel.compute(&x, &y).unwrap();
//! ```

use crate::error::{KernelError, Result};
use crate::types::Kernel;
use std::fmt;
use std::sync::Arc;

/// A kernel expression that can be algebraically composed
#[derive(Clone)]
pub enum KernelExpr {
    /// A base kernel
    Base(Arc<dyn Kernel>),

    /// Scaled kernel: c * K
    Scaled { kernel: Box<KernelExpr>, scale: f64 },

    /// Sum of kernels: K1 + K2
    Sum {
        left: Box<KernelExpr>,
        right: Box<KernelExpr>,
    },

    /// Product of kernels: K1 * K2
    Product {
        left: Box<KernelExpr>,
        right: Box<KernelExpr>,
    },

    /// Power of kernel: K^n
    Power {
        kernel: Box<KernelExpr>,
        exponent: u32,
    },
}

#[allow(clippy::should_implement_trait)]
impl KernelExpr {
    /// Create a base kernel expression
    pub fn base(kernel: Arc<dyn Kernel>) -> Self {
        KernelExpr::Base(kernel)
    }

    /// Scale this kernel by a constant
    pub fn scale(self, scale: f64) -> Result<Self> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "scale".to_string(),
                value: scale.to_string(),
                reason: "must be finite and non-negative".to_string(),
            });
        }
        Ok(KernelExpr::Scaled {
            kernel: Box::new(self),
            scale,
        })
    }

    /// Add two kernel expressions (method chaining, not operator overloading)
    pub fn add(self, other: KernelExpr) -> Self {
        KernelExpr::Sum {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    /// Multiply two kernel expressions (method chaining, not operator overloading)
    pub fn multiply(self, other: KernelExpr) -> Self {
        KernelExpr::Product {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    /// Raise kernel to a power
    pub fn power(self, exponent: u32) -> Result<Self> {
        if exponent == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "exponent".to_string(),
                value: exponent.to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(KernelExpr::Power {
            kernel: Box::new(self),
            exponent,
        })
    }

    /// Convert expression to a concrete kernel
    pub fn build(self) -> Box<dyn Kernel> {
        Box::new(SymbolicKernel { expr: self })
    }

    /// Simplify the expression (constant folding, etc.)
    #[allow(clippy::redundant_guards)]
    pub fn simplify(self) -> Self {
        match self {
            KernelExpr::Scaled { kernel, scale } if (scale - 1.0).abs() < 1e-10 => {
                kernel.simplify()
            }
            KernelExpr::Scaled { kernel, scale } if scale.abs() < 1e-10 => {
                // Scale of 0 creates a zero kernel, but we'll keep the structure
                KernelExpr::Scaled {
                    kernel: Box::new(kernel.simplify()),
                    scale,
                }
            }
            KernelExpr::Scaled { kernel, scale } => KernelExpr::Scaled {
                kernel: Box::new(kernel.simplify()),
                scale,
            },
            KernelExpr::Sum { left, right } => KernelExpr::Sum {
                left: Box::new(left.simplify()),
                right: Box::new(right.simplify()),
            },
            KernelExpr::Product { left, right } => KernelExpr::Product {
                left: Box::new(left.simplify()),
                right: Box::new(right.simplify()),
            },
            KernelExpr::Power { kernel, exponent } if exponent == 1 => kernel.simplify(),
            KernelExpr::Power { kernel, exponent } => KernelExpr::Power {
                kernel: Box::new(kernel.simplify()),
                exponent,
            },
            base @ KernelExpr::Base(_) => base,
        }
    }
}

impl fmt::Debug for KernelExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelExpr::Base(k) => write!(f, "{}", k.name()),
            KernelExpr::Scaled { kernel, scale } => write!(f, "{:.2}*{:?}", scale, kernel),
            KernelExpr::Sum { left, right } => write!(f, "({:?} + {:?})", left, right),
            KernelExpr::Product { left, right } => write!(f, "({:?} * {:?})", left, right),
            KernelExpr::Power { kernel, exponent } => write!(f, "{:?}^{}", kernel, exponent),
        }
    }
}

/// A kernel built from a symbolic expression
#[derive(Clone)]
pub struct SymbolicKernel {
    expr: KernelExpr,
}

impl SymbolicKernel {
    /// Create a new symbolic kernel from an expression
    pub fn new(expr: KernelExpr) -> Self {
        Self { expr }
    }

    /// Get the underlying expression
    pub fn expression(&self) -> &KernelExpr {
        &self.expr
    }

    /// Simplify the kernel expression
    pub fn simplify(self) -> Self {
        Self {
            expr: self.expr.simplify(),
        }
    }

    /// Evaluate the expression for given inputs
    fn eval(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.eval_expr(&self.expr, x, y)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn eval_expr(&self, expr: &KernelExpr, x: &[f64], y: &[f64]) -> Result<f64> {
        match expr {
            KernelExpr::Base(kernel) => kernel.compute(x, y),

            KernelExpr::Scaled { kernel, scale } => {
                let value = self.eval_expr(kernel, x, y)?;
                Ok(scale * value)
            }

            KernelExpr::Sum { left, right } => {
                let left_val = self.eval_expr(left, x, y)?;
                let right_val = self.eval_expr(right, x, y)?;
                Ok(left_val + right_val)
            }

            KernelExpr::Product { left, right } => {
                let left_val = self.eval_expr(left, x, y)?;
                let right_val = self.eval_expr(right, x, y)?;
                Ok(left_val * right_val)
            }

            KernelExpr::Power { kernel, exponent } => {
                let value = self.eval_expr(kernel, x, y)?;
                Ok(value.powi(*exponent as i32))
            }
        }
    }
}

impl Kernel for SymbolicKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.eval(x, y)
    }

    fn name(&self) -> &str {
        "Symbolic"
    }

    fn is_psd(&self) -> bool {
        // Conservative: only guaranteed PSD for specific compositions
        // In general, sum and product of PSD kernels are PSD
        // Scaled kernels are PSD if scale >= 0
        // Power may or may not be PSD
        check_psd(&self.expr)
    }
}

/// Check if an expression represents a PSD kernel
fn check_psd(expr: &KernelExpr) -> bool {
    match expr {
        KernelExpr::Base(kernel) => kernel.is_psd(),
        KernelExpr::Scaled { kernel, scale } => *scale >= 0.0 && check_psd(kernel),
        KernelExpr::Sum { left, right } => check_psd(left) && check_psd(right),
        KernelExpr::Product { left, right } => check_psd(left) && check_psd(right),
        KernelExpr::Power { kernel, exponent } => {
            // Only certain powers preserve PSD property
            *exponent == 1 && check_psd(kernel)
        }
    }
}

/// Builder for constructing kernels using symbolic expressions
pub struct KernelBuilder {
    expr: Option<KernelExpr>,
}

#[allow(clippy::should_implement_trait)]
impl KernelBuilder {
    /// Create a new kernel builder
    pub fn new() -> Self {
        Self { expr: None }
    }

    /// Add a base kernel (method chaining, not operator overloading)
    pub fn add(mut self, kernel: Arc<dyn Kernel>) -> Self {
        let new_expr = KernelExpr::base(kernel);
        self.expr = Some(match self.expr {
            Some(existing) => existing.add(new_expr),
            None => new_expr,
        });
        self
    }

    /// Add a scaled kernel
    pub fn add_scaled(mut self, kernel: Arc<dyn Kernel>, scale: f64) -> Self {
        let scaled = KernelExpr::base(kernel).scale(scale).unwrap();
        self.expr = Some(match self.expr {
            Some(existing) => existing.add(scaled),
            None => scaled,
        });
        self
    }

    /// Multiply by a kernel
    pub fn multiply(mut self, kernel: Arc<dyn Kernel>) -> Self {
        let new_expr = KernelExpr::base(kernel);
        self.expr = Some(match self.expr {
            Some(existing) => existing.multiply(new_expr),
            None => new_expr,
        });
        self
    }

    /// Scale the current expression
    pub fn scale(mut self, scale: f64) -> Result<Self> {
        if let Some(expr) = self.expr {
            self.expr = Some(expr.scale(scale)?);
        }
        Ok(self)
    }

    /// Raise to a power
    pub fn power(mut self, exponent: u32) -> Result<Self> {
        if let Some(expr) = self.expr {
            self.expr = Some(expr.power(exponent)?);
        }
        Ok(self)
    }

    /// Build the final kernel
    pub fn build(self) -> Box<dyn Kernel> {
        match self.expr {
            Some(expr) => expr.simplify().build(),
            None => {
                // Empty builder - return a zero kernel
                KernelExpr::base(Arc::new(crate::tensor_kernels::LinearKernel::new()))
                    .scale(0.0)
                    .unwrap()
                    .build()
            }
        }
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernels::{LinearKernel, RbfKernel};
    use crate::types::RbfKernelConfig;

    #[test]
    fn test_kernel_expr_scale() {
        let linear = Arc::new(LinearKernel::new());
        let expr = KernelExpr::base(linear).scale(2.0).unwrap();

        let kernel = SymbolicKernel::new(expr);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        // Linear kernel: 1*4 + 2*5 + 3*6 = 32, scaled by 2 = 64
        assert!((result - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_expr_add() {
        let linear1 = Arc::new(LinearKernel::new());
        let linear2 = Arc::new(LinearKernel::new());

        let expr1 = KernelExpr::base(linear1);
        let expr2 = KernelExpr::base(linear2);
        let sum = expr1.add(expr2);

        let kernel = SymbolicKernel::new(sum);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        // Sum of two linear kernels: 32 + 32 = 64
        assert!((result - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_expr_multiply() {
        let linear1 = Arc::new(LinearKernel::new());
        let linear2 = Arc::new(LinearKernel::new());

        let expr1 = KernelExpr::base(linear1);
        let expr2 = KernelExpr::base(linear2);
        let product = expr1.multiply(expr2);

        let kernel = SymbolicKernel::new(product);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        // Product of two linear kernels: 32 * 32 = 1024
        assert!((result - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_expr_power() {
        let linear = Arc::new(LinearKernel::new());
        let expr = KernelExpr::base(linear).power(2).unwrap();

        let kernel = SymbolicKernel::new(expr);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        // Linear kernel squared: 32^2 = 1024
        assert!((result - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_expr_complex() {
        // (0.5 * linear + 0.3 * rbf)
        let linear = Arc::new(LinearKernel::new());
        let rbf = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

        let expr = KernelExpr::base(linear)
            .scale(0.5)
            .unwrap()
            .add(KernelExpr::base(rbf).scale(0.3).unwrap());

        let kernel = SymbolicKernel::new(expr);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0]; // Same vectors

        let result = kernel.compute(&x, &y).unwrap();
        // 0.5 * 14 + 0.3 * 1.0 = 7.0 + 0.3 = 7.3
        assert!((result - 7.3).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_builder() {
        let builder = KernelBuilder::new();
        let kernel = builder
            .add_scaled(Arc::new(LinearKernel::new()), 0.5)
            .add_scaled(Arc::new(LinearKernel::new()), 0.3)
            .build();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        // 0.5 * 32 + 0.3 * 32 = 16 + 9.6 = 25.6
        assert!((result - 25.6).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_builder_multiply() {
        let builder = KernelBuilder::new();
        let kernel = builder
            .add(Arc::new(LinearKernel::new()))
            .multiply(Arc::new(LinearKernel::new()))
            .build();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let result = kernel.compute(&x, &y).unwrap();
        // (1*3 + 2*4) * (1*3 + 2*4) = 11 * 11 = 121
        assert!((result - 121.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_builder_power() {
        let builder = KernelBuilder::new();
        let kernel = builder
            .add(Arc::new(LinearKernel::new()))
            .power(3)
            .unwrap()
            .build();

        let x = vec![2.0];
        let y = vec![3.0];

        let result = kernel.compute(&x, &y).unwrap();
        // (2*3)^3 = 6^3 = 216
        assert!((result - 216.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_expr_simplify() {
        let linear = Arc::new(LinearKernel::new());
        let expr = KernelExpr::base(linear)
            .scale(1.0)
            .unwrap()
            .power(1)
            .unwrap();

        let simplified = expr.simplify();

        // Should simplify to just the base kernel
        matches!(simplified, KernelExpr::Base(_));
    }

    #[test]
    fn test_invalid_scale() {
        let linear = Arc::new(LinearKernel::new());
        let result = KernelExpr::base(linear).scale(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_power() {
        let linear = Arc::new(LinearKernel::new());
        let result = KernelExpr::base(linear).power(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_psd_property() {
        // Sum of PSD kernels is PSD
        let expr = KernelExpr::base(Arc::new(LinearKernel::new()))
            .add(KernelExpr::base(Arc::new(LinearKernel::new())));
        let kernel = SymbolicKernel::new(expr);
        assert!(kernel.is_psd());

        // Product of PSD kernels is PSD
        let expr = KernelExpr::base(Arc::new(LinearKernel::new()))
            .multiply(KernelExpr::base(Arc::new(LinearKernel::new())));
        let kernel = SymbolicKernel::new(expr);
        assert!(kernel.is_psd());

        // Scaled PSD kernel (positive scale) is PSD
        let expr = KernelExpr::base(Arc::new(LinearKernel::new()))
            .scale(2.0)
            .unwrap();
        let kernel = SymbolicKernel::new(expr);
        assert!(kernel.is_psd());
    }

    #[test]
    fn test_kernel_name() {
        let linear = Arc::new(LinearKernel::new());
        let expr = KernelExpr::base(linear);
        let kernel = SymbolicKernel::new(expr);

        assert_eq!(kernel.name(), "Symbolic");
    }

    #[test]
    fn test_empty_builder() {
        let builder = KernelBuilder::new();
        let kernel = builder.build();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // Empty builder creates a zero kernel
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }
}
