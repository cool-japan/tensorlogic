//! Advanced gradient operations for non-differentiable logical operations.
//!
//! This module provides differentiable approximations for operations that are
//! typically non-differentiable, enabling end-to-end training of logical models.
//!
//! ## Gradient Estimators
//!
//! - **Straight-Through Estimator (STE)**: Passes gradients through non-differentiable
//!   operations (thresholding, argmax, etc.) by treating them as identity in the
//!   backward pass.
//!
//! - **Gumbel-Softmax**: Continuous relaxation of categorical distributions,
//!   allowing differentiable sampling with temperature annealing.
//!
//! - **Soft Quantifiers**: Differentiable approximations of ∃ (exists) and ∀ (forall)
//!   using smooth max/min or probabilistic interpretations.

use crate::error::TlBackendResult;
use crate::Scirs2Tensor;
use scirs2_core::ndarray::{Array, ArrayD, Axis};
use scirs2_core::random::arrays::OptimizedArrayRandom;
use scirs2_core::random::prelude::*;

/// Straight-Through Estimator (STE) gradient configuration.
///
/// STE allows gradients to flow through non-differentiable operations
/// by using a different forward and backward pass:
/// - Forward: Apply the discrete/thresholded operation
/// - Backward: Pass gradients through as if it were identity
#[derive(Debug, Clone, Copy)]
pub struct SteConfig {
    /// Threshold for binarization (default: 0.5)
    pub threshold: f64,
    /// Whether to clip gradients to [-1, 1] range
    pub clip_gradients: bool,
}

impl Default for SteConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            clip_gradients: false,
        }
    }
}

/// Gumbel-Softmax configuration for differentiable categorical sampling.
///
/// Provides a continuous relaxation of categorical distributions using
/// Gumbel-Max trick combined with softmax temperature.
#[derive(Debug, Clone, Copy)]
pub struct GumbelSoftmaxConfig {
    /// Temperature parameter (τ): lower → harder, higher → softer
    /// Typical range: [0.1, 10.0], training starts high and anneals down
    pub temperature: f64,
    /// Whether to use hard (one-hot) samples in forward pass
    /// but soft samples for gradient computation (straight-through)
    pub hard: bool,
    /// Random seed for reproducibility (None = non-deterministic)
    pub seed: Option<u64>,
}

impl Default for GumbelSoftmaxConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            hard: false,
            seed: None,
        }
    }
}

/// Configuration for soft differentiable quantifiers.
///
/// Provides smooth approximations of logical quantifiers:
/// - ∃ (exists): At least one element is true
/// - ∀ (forall): All elements are true
#[derive(Debug, Clone, Copy)]
pub enum QuantifierMode {
    /// Hard quantifiers using max/min (non-differentiable)
    Hard,
    /// Soft using smooth approximations (log-sum-exp for max)
    Smooth { temperature: f64 },
    /// Probabilistic interpretation (1 - ∏(1-x) for ∃)
    Probabilistic,
}

impl Default for QuantifierMode {
    fn default() -> Self {
        Self::Smooth { temperature: 1.0 }
    }
}

/// Applies Straight-Through Estimator for binary thresholding.
///
/// Forward: y = (x >= threshold) ? 1.0 : 0.0
/// Backward: ∂L/∂x = ∂L/∂y (identity gradient)
///
/// # Arguments
/// * `input` - Input tensor with values in [0, 1]
/// * `config` - STE configuration
///
/// # Returns
/// Binarized tensor (forward pass result)
pub fn ste_threshold(input: &Scirs2Tensor, config: SteConfig) -> TlBackendResult<Scirs2Tensor> {
    // Forward: Apply threshold
    let output = input.mapv(|x| if x >= config.threshold { 1.0 } else { 0.0 });
    Ok(output)
}

/// Computes the backward pass gradient for STE threshold.
///
/// # Arguments
/// * `grad_output` - Gradient from downstream
/// * `input` - Original input tensor (unused in basic STE)
/// * `config` - STE configuration
///
/// # Returns
/// Gradient with respect to input (passed through)
pub fn ste_threshold_backward(
    grad_output: &Scirs2Tensor,
    _input: &Scirs2Tensor,
    config: SteConfig,
) -> TlBackendResult<Scirs2Tensor> {
    if config.clip_gradients {
        // Clip gradients to [-1, 1] range
        Ok(grad_output.mapv(|g| g.clamp(-1.0, 1.0)))
    } else {
        // Pass through gradient as-is
        Ok(grad_output.clone())
    }
}

/// Applies Gumbel-Softmax for differentiable categorical sampling.
///
/// Adds Gumbel noise to logits and applies temperature-scaled softmax:
/// y_i = exp((log(p_i) + g_i) / τ) / Σ_j exp((log(p_j) + g_j) / τ)
///
/// where g_i ~ Gumbel(0, 1) = -log(-log(U(0,1)))
///
/// # Arguments
/// * `logits` - Input logits (unnormalized log-probabilities)
/// * `config` - Gumbel-Softmax configuration
///
/// # Returns
/// Soft samples (or hard one-hot if config.hard = true)
pub fn gumbel_softmax(
    logits: &Scirs2Tensor,
    config: GumbelSoftmaxConfig,
) -> TlBackendResult<Scirs2Tensor> {
    // Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0, 1)
    let gumbel_noise = sample_gumbel(logits.shape(), config.seed)?;

    // Add noise to logits: logits + gumbel_noise
    let noisy_logits = logits + &gumbel_noise;

    // Apply temperature-scaled softmax
    let soft_samples = softmax_temperature(&noisy_logits, config.temperature)?;

    if config.hard {
        // Hard one-hot samples in forward, soft in backward (STE)
        let hard_samples = argmax_to_onehot(&soft_samples)?;
        Ok(hard_samples)
    } else {
        Ok(soft_samples)
    }
}

/// Computes gradient for Gumbel-Softmax backward pass.
///
/// For soft mode: Standard softmax gradient
/// For hard mode: Straight-through (gradient flows through soft samples)
///
/// # Arguments
/// * `grad_output` - Gradient from downstream
/// * `soft_samples` - Soft samples from forward pass
/// * `config` - Gumbel-Softmax configuration
///
/// # Returns
/// Gradient with respect to logits
pub fn gumbel_softmax_backward(
    grad_output: &Scirs2Tensor,
    soft_samples: &Scirs2Tensor,
    config: GumbelSoftmaxConfig,
) -> TlBackendResult<Scirs2Tensor> {
    // Softmax gradient: ∂L/∂logits = soft_samples * (∂L/∂y - (soft_samples · ∂L/∂y))
    // For hard mode, we use STE (pass through as if soft)

    // Compute dot product: sum(soft_samples * grad_output) along last axis
    let last_axis = soft_samples.ndim() - 1;
    let dot_product = (soft_samples * grad_output)
        .sum_axis(Axis(last_axis))
        .insert_axis(Axis(last_axis));

    // Compute gradient: soft_samples * (grad_output - dot_product)
    let grad_logits = soft_samples * &(grad_output - &dot_product);

    // Scale by temperature
    Ok(grad_logits.mapv(|g| g / config.temperature))
}

/// Applies soft exists quantifier: ∃x. P(x).
///
/// Differentiable approximation of "at least one element is true".
///
/// # Arguments
/// * `input` - Input tensor with values in [0, 1]
/// * `axis` - Axis along which to apply quantifier (None = all axes)
/// * `mode` - Quantifier mode (Hard/Smooth/Probabilistic)
///
/// # Returns
/// Result of exists quantification
pub fn soft_exists(
    input: &Scirs2Tensor,
    axis: Option<usize>,
    mode: QuantifierMode,
) -> TlBackendResult<Scirs2Tensor> {
    match mode {
        QuantifierMode::Hard => {
            // Hard max (non-differentiable)
            if let Some(ax) = axis {
                Ok(input.map_axis(Axis(ax), |slice| {
                    slice.iter().fold(0.0_f64, |a, &b| a.max(b))
                }))
            } else {
                let max_val = input.iter().fold(0.0_f64, |a, &b| a.max(b));
                Ok(Array::from_elem(vec![], max_val))
            }
        }
        QuantifierMode::Smooth { temperature } => {
            // Smooth max using log-sum-exp: max(x) ≈ τ * log(Σ exp(x/τ))
            smooth_max(input, axis, temperature)
        }
        QuantifierMode::Probabilistic => {
            // Probabilistic: 1 - ∏(1 - x_i)
            // This is equivalent to OR in probability theory
            probabilistic_exists(input, axis)
        }
    }
}

/// Computes gradient for soft exists quantifier.
///
/// # Arguments
/// * `grad_output` - Gradient from downstream
/// * `input` - Original input tensor
/// * `output` - Output from forward pass
/// * `axis` - Axis along which quantifier was applied
/// * `mode` - Quantifier mode
///
/// # Returns
/// Gradient with respect to input
pub fn soft_exists_backward(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    _output: &Scirs2Tensor,
    axis: Option<usize>,
    mode: QuantifierMode,
) -> TlBackendResult<Scirs2Tensor> {
    match mode {
        QuantifierMode::Hard => {
            // For hard max, gradient goes only to the maximum element
            // This is similar to argmax gradient (sparse)
            argmax_gradient(grad_output, input, axis)
        }
        QuantifierMode::Smooth { temperature } => {
            // Smooth max gradient: softmax weights
            smooth_max_gradient(grad_output, input, temperature, axis)
        }
        QuantifierMode::Probabilistic => {
            // Probabilistic gradient: ∂(1 - ∏(1-x_i))/∂x_j = ∏_{i≠j}(1-x_i)
            probabilistic_exists_gradient(grad_output, input, axis)
        }
    }
}

/// Applies soft forall quantifier: ∀x. P(x).
///
/// Differentiable approximation of "all elements are true".
///
/// # Arguments
/// * `input` - Input tensor with values in [0, 1]
/// * `axis` - Axis along which to apply quantifier (None = all axes)
/// * `mode` - Quantifier mode (Hard/Smooth/Probabilistic)
///
/// # Returns
/// Result of forall quantification
pub fn soft_forall(
    input: &Scirs2Tensor,
    axis: Option<usize>,
    mode: QuantifierMode,
) -> TlBackendResult<Scirs2Tensor> {
    // ∀x. P(x) is equivalent to ¬∃x. ¬P(x)
    // Or directly: min(x) in hard mode, product in probabilistic
    match mode {
        QuantifierMode::Hard => {
            // Hard min (non-differentiable)
            if let Some(ax) = axis {
                Ok(input.map_axis(Axis(ax), |slice| {
                    slice.iter().fold(1.0_f64, |a, &b| a.min(b))
                }))
            } else {
                let min_val = input.iter().fold(1.0_f64, |a, &b| a.min(b));
                Ok(Array::from_elem(vec![], min_val))
            }
        }
        QuantifierMode::Smooth { temperature } => {
            // Smooth min using -log-sum-exp(-x): min(x) ≈ -τ * log(Σ exp(-x/τ))
            smooth_min(input, axis, temperature)
        }
        QuantifierMode::Probabilistic => {
            // Probabilistic: ∏ x_i (product of probabilities)
            probabilistic_forall(input, axis)
        }
    }
}

/// Computes gradient for soft forall quantifier.
///
/// # Arguments
/// * `grad_output` - Gradient from downstream
/// * `input` - Original input tensor
/// * `output` - Output from forward pass
/// * `axis` - Axis along which quantifier was applied
/// * `mode` - Quantifier mode
///
/// # Returns
/// Gradient with respect to input
pub fn soft_forall_backward(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    output: &Scirs2Tensor,
    axis: Option<usize>,
    mode: QuantifierMode,
) -> TlBackendResult<Scirs2Tensor> {
    match mode {
        QuantifierMode::Hard => {
            // For hard min, gradient goes only to the minimum element
            argmin_gradient(grad_output, input, axis)
        }
        QuantifierMode::Smooth { temperature } => {
            // Smooth min gradient: similar to smooth max but with negated values
            smooth_min_gradient(grad_output, input, temperature, axis)
        }
        QuantifierMode::Probabilistic => {
            // Product gradient: ∂(∏ x_i)/∂x_j = (∏ x_i) / x_j
            probabilistic_forall_gradient(grad_output, input, output, axis)
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Samples Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0, 1).
fn sample_gumbel(shape: &[usize], seed: Option<u64>) -> TlBackendResult<Scirs2Tensor> {
    use scirs2_core::ndarray::IxDyn;

    let uniform_dist = Uniform::new(1e-10, 1.0 - 1e-10).unwrap(); // Avoid log(0)
    let dyn_shape = IxDyn(shape);

    let gumbel = if let Some(s) = seed {
        let mut rng = seeded_rng(s);
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    } else {
        let mut rng = thread_rng();
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    };

    // Apply Gumbel transformation: -log(-log(u))
    let gumbel = gumbel.mapv(|u: f64| -(-u.ln()).ln());
    Ok(gumbel)
}

/// Applies temperature-scaled softmax along the last axis.
fn softmax_temperature(logits: &Scirs2Tensor, temperature: f64) -> TlBackendResult<Scirs2Tensor> {
    // Scale by temperature
    let scaled = logits.mapv(|x| x / temperature);

    // Compute softmax along last axis
    let last_axis = scaled.ndim() - 1;

    // Subtract max for numerical stability
    let max_vals = scaled
        .map_axis(Axis(last_axis), |slice| {
            slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        })
        .insert_axis(Axis(last_axis));

    let exp_vals = (&scaled - &max_vals).mapv(|x| x.exp());
    let sum_exp = exp_vals
        .sum_axis(Axis(last_axis))
        .insert_axis(Axis(last_axis));

    Ok(&exp_vals / &sum_exp)
}

/// Converts soft samples to hard one-hot encoding (argmax).
fn argmax_to_onehot(soft_samples: &Scirs2Tensor) -> TlBackendResult<Scirs2Tensor> {
    let last_axis = soft_samples.ndim() - 1;
    let mut onehot = ArrayD::zeros(soft_samples.raw_dim());

    // Iterate over all elements except the last axis
    let n_classes = soft_samples.len_of(Axis(last_axis));

    // Get views along the last axis
    for i in 0..soft_samples.len() / n_classes {
        // Calculate multi-dimensional index
        let mut flat_idx = i;
        let mut indices = vec![0; soft_samples.ndim()];

        for dim in (0..last_axis).rev() {
            let size = soft_samples.len_of(Axis(dim));
            indices[dim] = flat_idx % size;
            flat_idx /= size;
        }

        // Find argmax along last dimension for this slice
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;

        for j in 0..n_classes {
            indices[last_axis] = j;
            let val = soft_samples[&indices[..]];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        // Set one-hot
        indices[last_axis] = max_idx;
        onehot[&indices[..]] = 1.0;
    }

    Ok(onehot)
}

/// Smooth max using log-sum-exp: max(x) ≈ τ * log(Σ exp(x/τ)).
fn smooth_max(
    input: &Scirs2Tensor,
    axis: Option<usize>,
    temperature: f64,
) -> TlBackendResult<Scirs2Tensor> {
    let scaled = input.mapv(|x| x / temperature);

    if let Some(ax) = axis {
        // Max for numerical stability
        let max_vals = scaled.map_axis(Axis(ax), |slice| {
            slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        });

        // For broadcasting, temporarily insert axis
        let max_vals_broadcast = max_vals.clone().insert_axis(Axis(ax));
        let exp_vals = (&scaled - &max_vals_broadcast).mapv(|x| x.exp());
        let sum_exp = exp_vals.sum_axis(Axis(ax));
        let log_sum_exp = &max_vals + &sum_exp.mapv(|x| x.ln());

        Ok(log_sum_exp.mapv(|x| x * temperature))
    } else {
        let max_val = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = scaled.mapv(|x| (x - max_val).exp());
        let sum_exp: f64 = exp_vals.iter().sum();
        let result = temperature * (max_val + sum_exp.ln());
        Ok(Array::from_elem(vec![], result))
    }
}

/// Gradient for smooth max.
fn smooth_max_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    temperature: f64,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    // Gradient: softmax weights
    let scaled = input.mapv(|x| x / temperature);

    if let Some(ax) = axis {
        let max_vals = scaled
            .map_axis(Axis(ax), |slice| {
                slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            })
            .insert_axis(Axis(ax));

        let exp_vals = (&scaled - &max_vals).mapv(|x| x.exp());
        let sum_exp = exp_vals.sum_axis(Axis(ax)).insert_axis(Axis(ax));
        let weights = &exp_vals / &sum_exp;

        // Broadcast grad_output and multiply
        Ok(&weights * grad_output)
    } else {
        let max_val = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = scaled.mapv(|x| (x - max_val).exp());
        let sum_exp: f64 = exp_vals.iter().sum();
        let weights = exp_vals.mapv(|x| x / sum_exp);

        let grad_scalar = grad_output.iter().next().unwrap_or(&0.0);
        Ok(weights.mapv(|w| w * grad_scalar))
    }
}

/// Smooth min using -log-sum-exp(-x).
fn smooth_min(
    input: &Scirs2Tensor,
    axis: Option<usize>,
    temperature: f64,
) -> TlBackendResult<Scirs2Tensor> {
    // min(x) = -max(-x)
    let negated = input.mapv(|x| -x);
    let result = smooth_max(&negated, axis, temperature)?;
    Ok(result.mapv(|x| -x))
}

/// Gradient for smooth min.
fn smooth_min_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    temperature: f64,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    // Same as smooth max gradient but with negated input
    let negated = input.mapv(|x| -x);
    let grad = smooth_max_gradient(grad_output, &negated, temperature, axis)?;
    Ok(grad.mapv(|g| -g))
}

/// Probabilistic exists: 1 - ∏(1 - x_i).
fn probabilistic_exists(
    input: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    let one_minus_input = input.mapv(|x| 1.0 - x);

    if let Some(ax) = axis {
        let product = one_minus_input.map_axis(Axis(ax), |slice| slice.iter().product::<f64>());
        Ok(product.mapv(|p| 1.0 - p))
    } else {
        let product: f64 = one_minus_input.iter().product();
        Ok(Array::from_elem(vec![], 1.0 - product))
    }
}

/// Gradient for probabilistic exists.
fn probabilistic_exists_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    // ∂(1 - ∏(1-x_i))/∂x_j = ∏_{i≠j}(1-x_i)
    let one_minus_input = input.mapv(|x| 1.0 - x);

    if let Some(ax) = axis {
        // For each element, compute product of all others
        let mut grad = ArrayD::zeros(input.raw_dim());

        for i in 0..input.len_of(Axis(ax)) {
            let mut slice = input.index_axis(Axis(ax), i).to_owned();
            // Product of all elements except i
            let product: f64 = one_minus_input.iter().product();
            let elem_val = 1.0 - input.index_axis(Axis(ax), i).iter().next().unwrap_or(&0.0);
            let grad_elem = if elem_val.abs() > 1e-10 {
                product / elem_val
            } else {
                0.0
            };

            slice.fill(grad_elem);
            grad.index_axis_mut(Axis(ax), i).assign(&slice);
        }

        Ok(&grad * grad_output)
    } else {
        let product: f64 = one_minus_input.iter().product();
        let grad = input.mapv(|x| {
            let denom = 1.0 - x;
            if denom.abs() > 1e-10 {
                product / denom
            } else {
                0.0
            }
        });

        let grad_scalar = grad_output.iter().next().unwrap_or(&0.0);
        Ok(grad.mapv(|g| g * grad_scalar))
    }
}

/// Probabilistic forall: ∏ x_i.
fn probabilistic_forall(
    input: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    if let Some(ax) = axis {
        Ok(input.map_axis(Axis(ax), |slice| slice.iter().product::<f64>()))
    } else {
        let product: f64 = input.iter().product();
        Ok(Array::from_elem(vec![], product))
    }
}

/// Gradient for probabilistic forall.
fn probabilistic_forall_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    output: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    // ∂(∏ x_i)/∂x_j = (∏ x_i) / x_j = output / x_j

    if let Some(_ax) = axis {
        // Broadcast output and divide by input
        let grad = output / input;
        Ok(&grad * grad_output)
    } else {
        let output_val = output.iter().next().unwrap_or(&0.0);
        let grad = input.mapv(|x| if x.abs() > 1e-10 { output_val / x } else { 0.0 });

        let grad_scalar = grad_output.iter().next().unwrap_or(&0.0);
        Ok(grad.mapv(|g| g * grad_scalar))
    }
}

/// Gradient for hard argmax (sparse gradient to maximum element).
fn argmax_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    let mut grad = ArrayD::zeros(input.raw_dim());

    if let Some(ax) = axis {
        for i in 0..input.len_of(Axis(ax)) {
            let slice = input.index_axis(Axis(ax), i);
            let max_idx = slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            grad.index_axis_mut(Axis(ax), i)[max_idx] = *grad_output
                .index_axis(Axis(ax), i)
                .iter()
                .next()
                .unwrap_or(&0.0);
        }
    } else {
        let max_idx = input
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        grad.as_slice_mut().unwrap()[max_idx] = *grad_output.iter().next().unwrap_or(&0.0);
    }

    Ok(grad)
}

/// Gradient for hard argmin (sparse gradient to minimum element).
fn argmin_gradient(
    grad_output: &Scirs2Tensor,
    input: &Scirs2Tensor,
    axis: Option<usize>,
) -> TlBackendResult<Scirs2Tensor> {
    let mut grad = ArrayD::zeros(input.raw_dim());

    if let Some(ax) = axis {
        for i in 0..input.len_of(Axis(ax)) {
            let slice = input.index_axis(Axis(ax), i);
            let min_idx = slice
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            grad.index_axis_mut(Axis(ax), i)[min_idx] = *grad_output
                .index_axis(Axis(ax), i)
                .iter()
                .next()
                .unwrap_or(&0.0);
        }
    } else {
        let min_idx = input
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        grad.as_slice_mut().unwrap()[min_idx] = *grad_output.iter().next().unwrap_or(&0.0);
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ste_threshold_forward() {
        let input = array![[0.2, 0.6], [0.4, 0.8]].into_dyn();
        let config = SteConfig::default();

        let output = ste_threshold(&input, config).unwrap();
        let expected = array![[0.0, 1.0], [0.0, 1.0]].into_dyn();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_ste_threshold_backward() {
        let grad_output = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let input = array![[0.2, 0.6], [0.4, 0.8]].into_dyn();
        let config = SteConfig::default();

        let grad_input = ste_threshold_backward(&grad_output, &input, config).unwrap();

        // Gradient passes through unchanged
        assert_eq!(grad_input, grad_output);
    }

    #[test]
    fn test_ste_gradient_clipping() {
        let grad_output = array![[5.0, -3.0], [0.5, -10.0]].into_dyn();
        let input = array![[0.2, 0.6], [0.4, 0.8]].into_dyn();
        let config = SteConfig {
            threshold: 0.5,
            clip_gradients: true,
        };

        let grad_input = ste_threshold_backward(&grad_output, &input, config).unwrap();
        let expected = array![[1.0, -1.0], [0.5, -1.0]].into_dyn();

        assert_eq!(grad_input, expected);
    }

    #[test]
    fn test_gumbel_softmax_deterministic() {
        let logits = array![[1.0, 2.0, 3.0]].into_dyn();
        let config = GumbelSoftmaxConfig {
            temperature: 1.0,
            hard: false,
            seed: Some(42),
        };

        let samples = gumbel_softmax(&logits, config).unwrap();

        // Check output is valid probability distribution
        assert_eq!(samples.shape(), &[1, 3]);
        let sum: f64 = samples.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be in [0, 1]
        for &val in samples.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_gumbel_softmax_hard_mode() {
        let logits = array![[1.0, 5.0, 2.0]].into_dyn();
        let config = GumbelSoftmaxConfig {
            temperature: 0.1,
            hard: true,
            seed: Some(123),
        };

        let samples = gumbel_softmax(&logits, config).unwrap();

        // In hard mode with low temperature and high logit[1], should be close to one-hot
        let sum: f64 = samples.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // At least one value should be close to 1.0
        let max_val = samples.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(max_val >= 0.9);
    }

    #[test]
    fn test_soft_exists_smooth() {
        let input = array![[0.1, 0.3], [0.2, 0.9]].into_dyn();
        let mode = QuantifierMode::Smooth { temperature: 1.0 };

        let output = soft_exists(&input, Some(1), mode).unwrap();

        // Should be approximately max along axis 1 (but higher due to log-sum-exp)
        // smooth_max([0.1, 0.3], τ=1) ≈ 0.898
        // smooth_max([0.2, 0.9], τ=1) ≈ 1.303
        assert_eq!(output.shape(), &[2]);
        assert!(
            output[0] >= 0.85 && output[0] <= 0.95,
            "output[0] = {} not in [0.85, 0.95]",
            output[0]
        );
        assert!(
            output[1] >= 1.25 && output[1] <= 1.35,
            "output[1] = {} not in [1.25, 1.35]",
            output[1]
        );
    }

    #[test]
    fn test_soft_exists_probabilistic() {
        let input = array![[0.5, 0.5]].into_dyn();
        let mode = QuantifierMode::Probabilistic;

        let output = soft_exists(&input, Some(1), mode).unwrap();

        // 1 - (1-0.5)*(1-0.5) = 1 - 0.25 = 0.75
        assert!((output[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_soft_forall_probabilistic() {
        let input = array![[0.5, 0.5]].into_dyn();
        let mode = QuantifierMode::Probabilistic;

        let output = soft_forall(&input, Some(1), mode).unwrap();

        // 0.5 * 0.5 = 0.25
        assert!((output[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_probabilistic_forall_gradient() {
        let input = array![[0.5, 0.8]].into_dyn();
        let output = array![0.4].into_dyn(); // 0.5 * 0.8
        let grad_output = array![1.0].into_dyn();

        let grad_input =
            probabilistic_forall_gradient(&grad_output, &input, &output, Some(1)).unwrap();

        // ∂(0.4)/∂x[0] = 0.4 / 0.5 = 0.8
        // ∂(0.4)/∂x[1] = 0.4 / 0.8 = 0.5
        assert!((grad_input[[0, 0]] - 0.8).abs() < 1e-6);
        assert!((grad_input[[0, 1]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_max_vs_hard_max() {
        let input = array![[1.0, 2.0, 3.0]].into_dyn();

        // Hard max
        let hard = soft_exists(&input, Some(1), QuantifierMode::Hard).unwrap();
        assert!((hard[0] - 3.0).abs() < 1e-6);

        // Smooth max with low temperature
        let smooth = soft_exists(
            &input,
            Some(1),
            QuantifierMode::Smooth { temperature: 0.01 },
        )
        .unwrap();
        assert!((smooth[0] - 3.0).abs() < 0.1); // Should be close to 3.0
    }

    #[test]
    fn test_gumbel_noise_properties() {
        // Test that Gumbel samples have correct properties
        let shape = &[1000];
        let noise = sample_gumbel(shape, Some(42)).unwrap();

        // Mean of Gumbel(0,1) is Euler-Mascheroni constant ≈ 0.5772
        let mean: f64 = noise.iter().sum::<f64>() / noise.len() as f64;
        assert!((mean - 0.5772).abs() < 0.1); // Rough check

        // Check no NaN or Inf
        for &val in noise.iter() {
            assert!(val.is_finite());
        }
    }
}
