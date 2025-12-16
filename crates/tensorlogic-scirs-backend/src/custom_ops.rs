//! Custom operations infrastructure with dynamic registration.
//!
//! This module provides a plugin system for user-defined tensor operations,
//! allowing extensibility without modifying the core executor.
//!
//! ## Features
//!
//! - **Custom Operation Trait**: Define operations with forward and backward passes
//! - **Operation Registry**: Dynamic registration and lookup
//! - **Gradient Support**: Automatic gradient computation for custom ops
//! - **Validation**: Shape and type checking for custom ops
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::custom_ops::{CustomOp, OpRegistry, CustomOpContext};
//! use tensorlogic_scirs_backend::Scirs2Tensor;
//!
//! // Define a custom softplus operation
//! struct SoftplusOp;
//!
//! impl CustomOp for SoftplusOp {
//!     fn name(&self) -> &str {
//!         "softplus"
//!     }
//!
//!     fn forward(&self, inputs: &[&Scirs2Tensor], _ctx: &mut CustomOpContext) -> Result<Scirs2Tensor, String> {
//!         let x = inputs[0];
//!         Ok(x.mapv(|v| (1.0 + v.exp()).ln()))
//!     }
//!
//!     fn backward(&self, grad: &Scirs2Tensor, inputs: &[&Scirs2Tensor], _ctx: &CustomOpContext) -> Result<Vec<Scirs2Tensor>, String> {
//!         let x = inputs[0];
//!         let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
//!         Ok(vec![grad * &sigmoid])
//!     }
//! }
//!
//! // Register and use
//! let mut registry = OpRegistry::new();
//! registry.register(Box::new(SoftplusOp));
//!
//! let result = registry.execute("softplus", &[&tensor], &mut context)?;
//! ```

use crate::{Scirs2Tensor, TlBackendError, TlBackendResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Context for custom operation execution.
///
/// Provides storage for intermediate values needed during backward pass
/// and metadata about the execution environment.
#[derive(Debug, Clone, Default)]
pub struct CustomOpContext {
    /// Storage for intermediate values (forward pass -> backward pass)
    pub intermediates: HashMap<String, Scirs2Tensor>,

    /// Custom metadata
    pub metadata: HashMap<String, String>,

    /// Whether gradient computation is enabled
    pub requires_grad: bool,
}

impl CustomOpContext {
    /// Create a new context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context with gradient computation enabled.
    pub fn with_grad() -> Self {
        Self {
            requires_grad: true,
            ..Default::default()
        }
    }

    /// Store an intermediate tensor for backward pass.
    pub fn save_for_backward(&mut self, name: impl Into<String>, tensor: Scirs2Tensor) {
        self.intermediates.insert(name.into(), tensor);
    }

    /// Retrieve a saved intermediate tensor.
    pub fn get_saved(&self, name: &str) -> Option<&Scirs2Tensor> {
        self.intermediates.get(name)
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Trait for custom tensor operations.
///
/// Implement this trait to define custom operations that can be registered
/// with the operation registry.
pub trait CustomOp: Send + Sync {
    /// Get the operation name.
    fn name(&self) -> &str;

    /// Number of inputs expected.
    fn num_inputs(&self) -> usize {
        1 // Default to unary operation
    }

    /// Execute the forward pass.
    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String>;

    /// Execute the backward pass (compute gradients).
    ///
    /// Returns gradients for each input tensor.
    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String>;

    /// Validate input shapes before execution.
    fn validate_inputs(&self, inputs: &[&Scirs2Tensor]) -> Result<(), String> {
        if inputs.len() != self.num_inputs() {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.num_inputs(),
                inputs.len()
            ));
        }
        Ok(())
    }

    /// Infer output shape from input shapes.
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, String> {
        // Default: same shape as first input
        if input_shapes.is_empty() {
            return Err("No input shapes provided".to_string());
        }
        Ok(input_shapes[0].to_vec())
    }
}

/// Registry for custom operations.
///
/// Manages registration and lookup of custom operations by name.
#[derive(Default)]
pub struct OpRegistry {
    /// Registered operations
    ops: HashMap<String, Arc<dyn CustomOp>>,
}

impl OpRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }

    /// Create a registry with common operations pre-registered.
    pub fn with_standard_ops() -> Self {
        let mut registry = Self::new();

        // Register common operations
        registry.register(Box::new(SoftplusOp));
        registry.register(Box::new(LeakyReluOp::default()));
        registry.register(Box::new(EluOp::default()));
        registry.register(Box::new(SwishOp));
        registry.register(Box::new(MishOp));
        registry.register(Box::new(GeluOp));
        registry.register(Box::new(HardSigmoidOp));
        registry.register(Box::new(HardSwishOp));

        registry
    }

    /// Register a custom operation.
    pub fn register(&mut self, op: Box<dyn CustomOp>) {
        self.ops.insert(op.name().to_string(), Arc::from(op));
    }

    /// Get a registered operation by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn CustomOp>> {
        self.ops.get(name).cloned()
    }

    /// Check if an operation is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.ops.contains_key(name)
    }

    /// List all registered operations.
    pub fn list_ops(&self) -> Vec<&str> {
        self.ops.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Execute a registered operation.
    pub fn execute(
        &self,
        name: &str,
        inputs: &[&Scirs2Tensor],
        ctx: &mut CustomOpContext,
    ) -> TlBackendResult<Scirs2Tensor> {
        let op = self
            .get(name)
            .ok_or_else(|| TlBackendError::unsupported(format!("Unknown operation: {}", name)))?;

        op.validate_inputs(inputs)
            .map_err(TlBackendError::execution)?;

        op.forward(inputs, ctx).map_err(TlBackendError::execution)
    }

    /// Execute backward pass for a registered operation.
    pub fn backward(
        &self,
        name: &str,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        ctx: &CustomOpContext,
    ) -> TlBackendResult<Vec<Scirs2Tensor>> {
        let op = self
            .get(name)
            .ok_or_else(|| TlBackendError::unsupported(format!("Unknown operation: {}", name)))?;

        op.backward(grad, inputs, ctx)
            .map_err(TlBackendError::gradient)
    }
}

// Standard custom operations

/// Softplus activation: ln(1 + exp(x))
pub struct SoftplusOp;

impl CustomOp for SoftplusOp {
    fn name(&self) -> &str {
        "softplus"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        // Numerically stable softplus
        Ok(x.mapv(|v| {
            if v > 20.0 {
                v // For large values, softplus ≈ x
            } else if v < -20.0 {
                v.exp() // For small values, softplus ≈ exp(x)
            } else {
                (1.0 + v.exp()).ln()
            }
        }))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        // d/dx softplus(x) = sigmoid(x)
        let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        Ok(vec![grad * &sigmoid])
    }
}

/// Leaky ReLU: max(alpha * x, x)
pub struct LeakyReluOp {
    /// Negative slope (default: 0.01)
    pub alpha: f64,
}

impl Default for LeakyReluOp {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl CustomOp for LeakyReluOp {
    fn name(&self) -> &str {
        "leaky_relu"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        let alpha = self.alpha;
        Ok(x.mapv(|v| if v > 0.0 { v } else { alpha * v }))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let alpha = self.alpha;
        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .map_collect(|&g, &v| if v > 0.0 { g } else { alpha * g });
        Ok(vec![grad_input])
    }
}

/// ELU: x if x > 0 else alpha * (exp(x) - 1)
pub struct EluOp {
    /// Scale for negative values (default: 1.0)
    pub alpha: f64,
}

impl Default for EluOp {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl CustomOp for EluOp {
    fn name(&self) -> &str {
        "elu"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        let alpha = self.alpha;
        let result = x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) });

        // Save for backward
        if ctx.requires_grad {
            ctx.save_for_backward("output", result.clone());
        }

        Ok(result)
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let alpha = self.alpha;

        let grad_input = if let Some(output) = ctx.get_saved("output") {
            // Use saved output for efficiency
            scirs2_core::ndarray::Zip::from(grad)
                .and(x)
                .and(output)
                .map_collect(|&g, &v, &o| if v > 0.0 { g } else { g * (o + alpha) })
        } else {
            // Compute from inputs
            scirs2_core::ndarray::Zip::from(grad)
                .and(x)
                .map_collect(|&g, &v| if v > 0.0 { g } else { g * alpha * v.exp() })
        };

        Ok(vec![grad_input])
    }
}

/// Swish: x * sigmoid(x)
pub struct SwishOp;

impl CustomOp for SwishOp {
    fn name(&self) -> &str {
        "swish"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let result = x * &sigmoid;

        if ctx.requires_grad {
            ctx.save_for_backward("sigmoid", sigmoid);
        }

        Ok(result)
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];

        let sigmoid = if let Some(s) = ctx.get_saved("sigmoid") {
            s.clone()
        } else {
            x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
        };

        // d/dx swish(x) = swish(x) + sigmoid(x) * (1 - swish(x))
        // = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .and(&sigmoid)
            .map_collect(|&g, &v, &s| g * (s + v * s * (1.0 - s)));

        Ok(vec![grad_input])
    }
}

/// Mish: x * tanh(softplus(x))
pub struct MishOp;

impl CustomOp for MishOp {
    fn name(&self) -> &str {
        "mish"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        Ok(x.mapv(|v| {
            let softplus = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            v * softplus.tanh()
        }))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        // Numerical gradient computation for mish
        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .map_collect(|&g, &v| {
                let e = v.exp();
                let omega = 4.0 * (v + 1.0) + 4.0 * e * e + e * e * e + e * (4.0 * v + 6.0);
                let delta = 2.0 * e + e * e + 2.0;
                g * e * omega / (delta * delta)
            });

        Ok(vec![grad_input])
    }
}

/// GELU: Gaussian Error Linear Unit
pub struct GeluOp;

impl CustomOp for GeluOp {
    fn name(&self) -> &str {
        "gelu"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        Ok(x.mapv(|v| {
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        }))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .map_collect(|&g, &v| {
                let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * v * v);

                g * (0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner)
            });

        Ok(vec![grad_input])
    }
}

/// Hard Sigmoid: clip((x + 3) / 6, 0, 1)
pub struct HardSigmoidOp;

impl CustomOp for HardSigmoidOp {
    fn name(&self) -> &str {
        "hard_sigmoid"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        Ok(x.mapv(|v| ((v + 3.0) / 6.0).clamp(0.0, 1.0)))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .map_collect(|&g, &v| if v > -3.0 && v < 3.0 { g / 6.0 } else { 0.0 });

        Ok(vec![grad_input])
    }
}

/// Hard Swish: x * hard_sigmoid(x)
pub struct HardSwishOp;

impl CustomOp for HardSwishOp {
    fn name(&self) -> &str {
        "hard_swish"
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        Ok(x.mapv(|v| {
            let hard_sigmoid = ((v + 3.0) / 6.0).clamp(0.0, 1.0);
            v * hard_sigmoid
        }))
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let grad_input = scirs2_core::ndarray::Zip::from(grad)
            .and(x)
            .map_collect(|&g, &v| {
                if v <= -3.0 {
                    0.0
                } else if v >= 3.0 {
                    g
                } else {
                    g * (v / 3.0 + 0.5)
                }
            });

        Ok(vec![grad_input])
    }
}

/// Binary custom operation for element-wise operations
pub struct BinaryCustomOp<F, G>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> (f64, f64) + Send + Sync,
{
    name: String,
    forward_fn: F,
    backward_fn: G,
}

impl<F, G> BinaryCustomOp<F, G>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> (f64, f64) + Send + Sync,
{
    /// Create a new binary custom operation.
    pub fn new(name: impl Into<String>, forward_fn: F, backward_fn: G) -> Self {
        Self {
            name: name.into(),
            forward_fn,
            backward_fn,
        }
    }
}

impl<F, G> CustomOp for BinaryCustomOp<F, G>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> (f64, f64) + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(
        &self,
        inputs: &[&Scirs2Tensor],
        _ctx: &mut CustomOpContext,
    ) -> Result<Scirs2Tensor, String> {
        let x = inputs[0];
        let y = inputs[1];

        if x.shape() != y.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }

        let result = scirs2_core::ndarray::Zip::from(x)
            .and(y)
            .map_collect(|&a, &b| (self.forward_fn)(a, b));

        Ok(result)
    }

    fn backward(
        &self,
        grad: &Scirs2Tensor,
        inputs: &[&Scirs2Tensor],
        _ctx: &CustomOpContext,
    ) -> Result<Vec<Scirs2Tensor>, String> {
        let x = inputs[0];
        let y = inputs[1];

        let mut grad_x = Scirs2Tensor::zeros(x.raw_dim());
        let mut grad_y = Scirs2Tensor::zeros(y.raw_dim());

        scirs2_core::ndarray::Zip::from(&mut grad_x)
            .and(&mut grad_y)
            .and(grad)
            .and(x)
            .and(y)
            .for_each(|gx, gy, &g, &a, &b| {
                let (dx, dy) = (self.backward_fn)(a, b, g);
                *gx = dx;
                *gy = dy;
            });

        Ok(vec![grad_x, grad_y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn create_tensor(data: Vec<f64>, shape: Vec<usize>) -> Scirs2Tensor {
        ArrayD::from_shape_vec(shape, data).unwrap()
    }

    #[test]
    fn test_op_registry_basic() {
        let mut registry = OpRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(SoftplusOp));
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("softplus"));
        assert!(!registry.contains("unknown"));
    }

    #[test]
    fn test_op_registry_with_standard_ops() {
        let registry = OpRegistry::with_standard_ops();
        assert!(registry.contains("softplus"));
        assert!(registry.contains("leaky_relu"));
        assert!(registry.contains("elu"));
        assert!(registry.contains("swish"));
        assert!(registry.contains("mish"));
        assert!(registry.contains("gelu"));
        assert!(registry.contains("hard_sigmoid"));
        assert!(registry.contains("hard_swish"));
    }

    #[test]
    fn test_softplus_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-1.0, 0.0, 1.0], vec![3]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("softplus", &[&tensor], &mut ctx).unwrap();

        // softplus(-1) ≈ 0.3133, softplus(0) = ln(2) ≈ 0.6931, softplus(1) ≈ 1.3133
        assert!(result[[0]] > 0.3 && result[[0]] < 0.35);
        assert!((result[[1]] - std::f64::consts::LN_2).abs() < 0.01);
        assert!(result[[2]] > 1.3 && result[[2]] < 1.35);
    }

    #[test]
    fn test_softplus_backward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![0.0], vec![1]);
        let grad = create_tensor(vec![1.0], vec![1]);
        let ctx = CustomOpContext::new();

        let grads = registry
            .backward("softplus", &grad, &[&tensor], &ctx)
            .unwrap();

        // d/dx softplus(0) = sigmoid(0) = 0.5
        assert!((grads[0][[0]] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-2.0, 0.0, 2.0], vec![3]);
        let mut ctx = CustomOpContext::new();

        let result = registry
            .execute("leaky_relu", &[&tensor], &mut ctx)
            .unwrap();

        assert!((result[[0]] - (-0.02)).abs() < 0.001); // -2 * 0.01
        assert_eq!(result[[1]], 0.0);
        assert_eq!(result[[2]], 2.0);
    }

    #[test]
    fn test_elu_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-1.0, 0.0, 1.0], vec![3]);
        let mut ctx = CustomOpContext::with_grad();

        let result = registry.execute("elu", &[&tensor], &mut ctx).unwrap();

        // elu(-1) = exp(-1) - 1 ≈ -0.632
        assert!((result[[0]] - (-0.632)).abs() < 0.01);
        assert_eq!(result[[1]], 0.0);
        assert_eq!(result[[2]], 1.0);
    }

    #[test]
    fn test_swish_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![0.0], vec![1]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("swish", &[&tensor], &mut ctx).unwrap();

        // swish(0) = 0 * 0.5 = 0
        assert_eq!(result[[0]], 0.0);
    }

    #[test]
    fn test_gelu_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-1.0, 0.0, 1.0], vec![3]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("gelu", &[&tensor], &mut ctx).unwrap();

        // gelu(0) = 0
        assert!((result[[1]]).abs() < 0.01);
        // gelu(x) has specific values
        assert!(result[[0]] < 0.0); // gelu(-1) is negative
        assert!(result[[2]] > 0.5); // gelu(1) > 0.5
    }

    #[test]
    fn test_hard_sigmoid_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-4.0, 0.0, 4.0], vec![3]);
        let mut ctx = CustomOpContext::new();

        let result = registry
            .execute("hard_sigmoid", &[&tensor], &mut ctx)
            .unwrap();

        assert_eq!(result[[0]], 0.0); // Clipped to 0
        assert_eq!(result[[1]], 0.5); // (0 + 3) / 6 = 0.5
        assert_eq!(result[[2]], 1.0); // Clipped to 1
    }

    #[test]
    fn test_hard_swish_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![-4.0, 0.0, 4.0], vec![3]);
        let mut ctx = CustomOpContext::new();

        let result = registry
            .execute("hard_swish", &[&tensor], &mut ctx)
            .unwrap();

        assert_eq!(result[[0]], 0.0); // -4 * 0 = 0
        assert_eq!(result[[1]], 0.0); // 0 * 0.5 = 0
        assert_eq!(result[[2]], 4.0); // 4 * 1 = 4
    }

    #[test]
    fn test_custom_op_context() {
        let mut ctx = CustomOpContext::with_grad();
        assert!(ctx.requires_grad);

        let tensor = create_tensor(vec![1.0, 2.0], vec![2]);
        ctx.save_for_backward("test", tensor.clone());

        let saved = ctx.get_saved("test").unwrap();
        assert_eq!(saved[[0]], 1.0);
        assert_eq!(saved[[1]], 2.0);

        ctx.set_metadata("key", "value");
        assert_eq!(ctx.get_metadata("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_binary_custom_op() {
        // Define a custom power operation
        let pow_op = BinaryCustomOp::new(
            "pow",
            |a, b| a.powf(b),
            |a, b, g| {
                let da = g * b * a.powf(b - 1.0);
                let db = g * a.powf(b) * a.ln();
                (da, db)
            },
        );

        let mut registry = OpRegistry::new();
        registry.register(Box::new(pow_op));

        let x = create_tensor(vec![2.0, 3.0], vec![2]);
        let y = create_tensor(vec![3.0, 2.0], vec![2]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("pow", &[&x, &y], &mut ctx).unwrap();

        assert_eq!(result[[0]], 8.0); // 2^3
        assert_eq!(result[[1]], 9.0); // 3^2
    }

    #[test]
    fn test_validate_inputs() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![1.0], vec![1]);
        let mut ctx = CustomOpContext::new();

        // Correct number of inputs
        let result = registry.execute("softplus", &[&tensor], &mut ctx);
        assert!(result.is_ok());

        // Wrong number of inputs
        let result = registry.execute("softplus", &[&tensor, &tensor], &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_ops() {
        let registry = OpRegistry::with_standard_ops();
        let ops = registry.list_ops();

        assert!(ops.contains(&"softplus"));
        assert!(ops.contains(&"gelu"));
    }

    #[test]
    fn test_unknown_operation() {
        let registry = OpRegistry::new();
        let tensor = create_tensor(vec![1.0], vec![1]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("unknown", &[&tensor], &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_mish_forward() {
        let registry = OpRegistry::with_standard_ops();
        let tensor = create_tensor(vec![0.0], vec![1]);
        let mut ctx = CustomOpContext::new();

        let result = registry.execute("mish", &[&tensor], &mut ctx).unwrap();

        // mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ≈ 0
        assert!(result[[0]].abs() < 0.01);
    }
}
