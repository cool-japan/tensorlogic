//! Optimizer wrappers around SciRS2 optimizers.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Compute the global L2 norm of all gradients.
///
/// # Arguments
/// * `gradients` - Gradients for all parameters
///
/// # Returns
/// The L2 norm of all gradients combined
fn compute_gradient_norm(gradients: &HashMap<String, Array<f64, Ix2>>) -> f64 {
    let mut total_norm_sq = 0.0;

    for grad in gradients.values() {
        for &g in grad.iter() {
            total_norm_sq += g * g;
        }
    }

    total_norm_sq.sqrt()
}

/// Gradient clipping mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradClipMode {
    /// Clip by value (element-wise).
    Value,
    /// Clip by global L2 norm.
    Norm,
}

/// Configuration for optimizers.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Momentum (for SGD).
    pub momentum: f64,
    /// Beta1 (for Adam/AdamW).
    pub beta1: f64,
    /// Beta2 (for Adam/AdamW).
    pub beta2: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
    /// Weight decay (for AdamW).
    pub weight_decay: f64,
    /// Gradient clipping threshold (None = no clipping).
    pub grad_clip: Option<f64>,
    /// Gradient clipping mode.
    pub grad_clip_mode: GradClipMode,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            grad_clip: None,
            grad_clip_mode: GradClipMode::Value,
        }
    }
}

/// Trait for optimizers.
pub trait Optimizer {
    /// Update parameters with computed gradients.
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()>;

    /// Zero all gradients.
    fn zero_grad(&mut self);

    /// Get current learning rate.
    fn get_lr(&self) -> f64;

    /// Set learning rate.
    fn set_lr(&mut self, lr: f64);

    /// Get optimizer state for checkpointing.
    fn state_dict(&self) -> HashMap<String, Vec<f64>>;

    /// Load optimizer state from checkpoint.
    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>);
}

/// SGD optimizer with momentum.
#[derive(Debug)]
pub struct SgdOptimizer {
    config: OptimizerConfig,
    /// Momentum buffers for each parameter.
    velocity: HashMap<String, Array<f64, Ix2>>,
}

impl SgdOptimizer {
    /// Create a new SGD optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    // Clip by value (element-wise)
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    // Clip by global L2 norm
                    let total_norm = compute_gradient_norm(gradients);

                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for SgdOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize velocity if not present
            if !self.velocity.contains_key(name) {
                self.velocity
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let velocity = self.velocity.get_mut(name).unwrap();

            // Update velocity: v = momentum * v + lr * grad
            velocity.mapv_inplace(|v| self.config.momentum * v);
            *velocity = &*velocity + &(grad * self.config.learning_rate);

            // Update parameter: param = param - velocity
            *param = &*param - &*velocity;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally, nothing to do here
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        for (name, velocity) in &self.velocity {
            state.insert(
                format!("velocity_{}", name),
                velocity.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("velocity_") {
                // Reconstruct array from values (assumes correct shape)
                if let Some(velocity) = self.velocity.get(name) {
                    let shape = velocity.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.velocity.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// Adam optimizer.
#[derive(Debug)]
pub struct AdamOptimizer {
    config: OptimizerConfig,
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates (exponential moving average of squared gradients).
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    // Clip by value (element-wise)
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    // Clip by global L2 norm
                    let total_norm = compute_gradient_norm(gradients);

                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;

        // Bias correction
        let lr_t =
            lr * ((1.0 - beta2.powi(self.t as i32)).sqrt()) / (1.0 - beta1.powi(self.t as i32));

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));

            // Update parameter: param = param - lr_t * m / (sqrt(v) + eps)
            let update = m.mapv(|m_val| m_val * lr_t) / &v.mapv(|v_val| v_val.sqrt() + eps);
            *param = &*param - &update;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally, nothing to do here
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
#[derive(Debug)]
pub struct AdamWOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    // Clip by value (element-wise)
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    // Clip by global L2 norm
                    let total_norm = compute_gradient_norm(gradients);

                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let weight_decay = self.config.weight_decay;

        // Bias correction
        let lr_t =
            lr * ((1.0 - beta2.powi(self.t as i32)).sqrt()) / (1.0 - beta1.powi(self.t as i32));

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();

            // Update biased first moment estimate
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update biased second raw moment estimate
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));

            // Compute Adam update
            let update = m.mapv(|m_val| m_val * lr_t) / &v.mapv(|v_val| v_val.sqrt() + eps);

            // Apply weight decay (decoupled from gradient)
            let decay = param.mapv(|p| p * lr * weight_decay);

            // Update parameter: param = param - update - decay
            *param = &*param - &update - &decay;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// RMSprop optimizer (Root Mean Square Propagation).
#[derive(Debug)]
pub struct RMSpropOptimizer {
    config: OptimizerConfig,
    /// Moving average of squared gradients.
    v: HashMap<String, Array<f64, Ix2>>,
}

impl RMSpropOptimizer {
    /// Create a new RMSprop optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            v: HashMap::new(),
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for RMSpropOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        let lr = self.config.learning_rate;
        let alpha = self.config.beta2; // Use beta2 as decay rate for RMSprop
        let eps = self.config.epsilon;

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moving average if not present
            if !self.v.contains_key(name) {
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let v = self.v.get_mut(name).unwrap();

            // Update moving average: v = alpha * v + (1 - alpha) * grad^2
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * alpha + &(grad_squared * (1.0 - alpha));

            // Update parameter: param = param - lr * grad / (sqrt(v) + eps)
            let update = grad / &v.mapv(|v_val| v_val.sqrt() + eps);
            *param = &*param - &(update * lr);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// Adagrad optimizer (Adaptive Gradient).
#[derive(Debug)]
pub struct AdagradOptimizer {
    config: OptimizerConfig,
    /// Accumulated sum of squared gradients.
    sum_squared_grads: HashMap<String, Array<f64, Ix2>>,
}

impl AdagradOptimizer {
    /// Create a new Adagrad optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            sum_squared_grads: HashMap::new(),
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for AdagradOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize accumulated sum if not present
            if !self.sum_squared_grads.contains_key(name) {
                self.sum_squared_grads
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let sum_sq = self.sum_squared_grads.get_mut(name).unwrap();

            // Accumulate squared gradients: sum_sq = sum_sq + grad^2
            let grad_squared = grad.mapv(|g| g * g);
            *sum_sq = &*sum_sq + &grad_squared;

            // Update parameter: param = param - lr * grad / (sqrt(sum_sq) + eps)
            let update = grad / &sum_sq.mapv(|s| s.sqrt() + eps);
            *param = &*param - &(update * lr);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        for (name, sum_sq) in &self.sum_squared_grads {
            state.insert(
                format!("sum_squared_grads_{}", name),
                sum_sq.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("sum_squared_grads_") {
                if let Some(sum_sq) = self.sum_squared_grads.get(name) {
                    let shape = sum_sq.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.sum_squared_grads.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// NAdam optimizer (Nesterov-accelerated Adam).
#[derive(Debug)]
pub struct NAdamOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl NAdamOptimizer {
    /// Create a new NAdam optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for NAdamOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;

        // Momentum schedule (schedule multiplier for beta1)
        let mu_t = beta1 * (1.0 - 0.5 * 0.96_f64.powi(self.t as i32));
        let mu_t_next = beta1 * (1.0 - 0.5 * 0.96_f64.powi((self.t + 1) as i32));

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();

            // Update biased first moment estimate
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update biased second moment estimate
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));

            // Bias correction
            let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));

            // Nesterov momentum
            let m_bar =
                &m_hat * mu_t_next / (1.0 - mu_t_next) + &(grad * (1.0 - mu_t) / (1.0 - mu_t_next));

            // Update parameter
            let update = m_bar / &v_hat.mapv(|v_val| v_val.sqrt() + eps);
            *param = &*param - &(update * lr);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training).
/// Designed for large batch training, uses layer-wise adaptation.
#[derive(Debug)]
pub struct LambOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl LambOptimizer {
    /// Create a new LAMB optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }

    /// Compute L2 norm of an array.
    fn compute_norm(arr: &Array<f64, Ix2>) -> f64 {
        arr.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl Optimizer for LambOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let weight_decay = self.config.weight_decay;

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();

            // Update biased first moment estimate
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update biased second moment estimate
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));

            // Bias correction
            let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));

            // Compute Adam step (without weight decay)
            let adam_step = &m_hat / &v_hat.mapv(|v_val| v_val.sqrt() + eps);

            // Add weight decay
            let update = &adam_step + &param.mapv(|p| p * weight_decay);

            // Layer-wise adaptation: compute trust ratio
            let param_norm = Self::compute_norm(param);
            let update_norm = Self::compute_norm(&update);

            let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
                param_norm / update_norm
            } else {
                1.0
            };

            // Update parameter with layer-wise adapted learning rate
            *param = &*param - &(update * (lr * trust_ratio));
        }

        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// AdaMax optimizer (variant of Adam with infinity norm).
///
/// Uses the infinity norm of gradients instead of L2 norm, making it more robust
/// to large gradients and outliers.
///
/// Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
#[derive(Debug)]
pub struct AdaMaxOptimizer {
    config: OptimizerConfig,
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Array<f64, Ix2>>,
    /// Exponentially weighted infinity norm.
    u: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdaMaxOptimizer {
    /// Create a new AdaMax optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            u: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for AdaMaxOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.u.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let u = self.u.get_mut(name).unwrap();

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update exponentially weighted infinity norm: u = max(beta2 * u, |grad|)
            for i in 0..u.nrows() {
                for j in 0..u.ncols() {
                    u[[i, j]] = (beta2 * u[[i, j]]).max(grad[[i, j]].abs());
                }
            }

            // Bias correction for first moment
            let bias_correction = 1.0 - beta1.powi(self.t as i32);
            let lr_t = lr / bias_correction;

            // Update parameter: param = param - lr_t * m / u
            for i in 0..param.nrows() {
                for j in 0..param.ncols() {
                    let update = lr_t * m[[i, j]] / (u[[i, j]] + self.config.epsilon);
                    param[[i, j]] -= update;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, u_val) in &self.u {
            state.insert(format!("u_{}", name), u_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("u_") {
                if let Some(u) = self.u.get(name) {
                    let shape = u.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.u.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// Lookahead optimizer (wrapper that uses slow and fast weights).
///
/// Maintains two sets of weights: fast weights updated by an inner optimizer,
/// and slow weights that are periodically updated as an exponential moving average.
///
/// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019
#[derive(Debug)]
pub struct LookaheadOptimizer<O: Optimizer> {
    /// Inner optimizer for fast weights.
    inner_optimizer: O,
    /// Slow weights (maintained separately).
    slow_weights: HashMap<String, Array<f64, Ix2>>,
    /// Interpolation coefficient (typically 0.5).
    alpha: f64,
    /// Number of inner optimizer steps before synchronization.
    k: usize,
    /// Current step counter.
    step_counter: usize,
}

impl<O: Optimizer> LookaheadOptimizer<O> {
    /// Create a new Lookahead optimizer.
    ///
    /// # Arguments
    /// * `inner_optimizer` - The inner optimizer (e.g., Adam, SGD)
    /// * `alpha` - Interpolation coefficient for slow weight update (typically 0.5)
    /// * `k` - Number of fast updates before slow weight synchronization (typically 5-10)
    pub fn new(inner_optimizer: O, alpha: f64, k: usize) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TrainError::InvalidParameter(
                "alpha must be in [0, 1]".to_string(),
            ));
        }
        if k == 0 {
            return Err(TrainError::InvalidParameter(
                "k must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            inner_optimizer,
            slow_weights: HashMap::new(),
            alpha,
            k,
            step_counter: 0,
        })
    }

    /// Initialize slow weights from current parameters.
    fn initialize_slow_weights(&mut self, parameters: &HashMap<String, Array<f64, Ix2>>) {
        if self.slow_weights.is_empty() {
            for (name, param) in parameters {
                self.slow_weights.insert(name.clone(), param.clone());
            }
        }
    }

    /// Synchronize slow weights with fast weights.
    fn synchronize_weights(&mut self, parameters: &mut HashMap<String, Array<f64, Ix2>>) {
        for (name, param) in parameters.iter_mut() {
            if let Some(slow_weight) = self.slow_weights.get_mut(name) {
                // Update slow weights: slow = slow + alpha * (fast - slow)
                *slow_weight = &*slow_weight + &((&*param - &*slow_weight) * self.alpha);

                // Update fast weights to slow weights
                *param = slow_weight.clone();
            }
        }
    }
}

impl<O: Optimizer> Optimizer for LookaheadOptimizer<O> {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        // Initialize slow weights on first step
        self.initialize_slow_weights(parameters);

        // Perform fast weight update using inner optimizer
        self.inner_optimizer.step(parameters, gradients)?;

        self.step_counter += 1;

        // Synchronize every k steps
        if self.step_counter.is_multiple_of(self.k) {
            self.synchronize_weights(parameters);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.inner_optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.inner_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = self.inner_optimizer.state_dict();

        // Add lookahead-specific state
        state.insert("step_counter".to_string(), vec![self.step_counter as f64]);
        state.insert("alpha".to_string(), vec![self.alpha]);
        state.insert("k".to_string(), vec![self.k as f64]);

        for (name, slow_weight) in &self.slow_weights {
            state.insert(
                format!("slow_{}", name),
                slow_weight.iter().copied().collect(),
            );
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        // Load inner optimizer state
        self.inner_optimizer.load_state_dict(state.clone());

        // Load lookahead-specific state
        if let Some(counter) = state.get("step_counter") {
            self.step_counter = counter[0] as usize;
        }
        if let Some(alpha_val) = state.get("alpha") {
            self.alpha = alpha_val[0];
        }
        if let Some(k_val) = state.get("k") {
            self.k = k_val[0] as usize;
        }

        // Load slow weights
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("slow_") {
                if let Some(slow_weight) = self.slow_weights.get(name) {
                    let shape = slow_weight.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.slow_weights.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// AdaBelief optimizer (NeurIPS 2020).
///
/// AdaBelief adapts the step size according to the "belief" in the gradient direction.
/// It uses the variance of gradients (belief) to adapt the learning rate, which can
/// achieve faster convergence and better generalization than Adam/AdamW.
///
/// Reference: Zhuang et al. "AdaBelief Optimizer: Adapting Stepsizes by the Belief
/// in Observed Gradients" (NeurIPS 2020)
#[derive(Debug)]
pub struct AdaBeliefOptimizer {
    config: OptimizerConfig,
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates (variance of gradients).
    s: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdaBeliefOptimizer {
    /// Create a new AdaBelief optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            s: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }
}

impl Optimizer for AdaBeliefOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let weight_decay = self.config.weight_decay;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.s.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let s = self.s.get_mut(name).unwrap();

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Compute gradient prediction error: (grad - m)
            let grad_diff = grad - &*m;

            // Update second moment (variance): s = beta2 * s + (1 - beta2) * (grad - m)^2
            let grad_diff_squared = grad_diff.mapv(|g| g * g);
            *s = &*s * beta2 + &(grad_diff_squared * (1.0 - beta2));

            // Bias-corrected moments
            let m_hat = &*m / bias_correction1;
            let s_hat = &*s / bias_correction2;

            // Weight decay (AdamW-style decoupled weight decay)
            if weight_decay > 0.0 {
                param.mapv_inplace(|p| p * (1.0 - lr * weight_decay));
            }

            // Update parameter: param = param - lr * m_hat / (sqrt(s_hat) + eps)
            let update = m_hat / (s_hat.mapv(|v| v.sqrt()) + eps);
            *param = &*param - &(update * lr);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, s_val) in &self.s {
            state.insert(format!("s_{}", name), s_val.iter().copied().collect());
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_val) = state.get("t") {
            self.t = t_val[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m_array) = self.m.get(name) {
                    let shape = m_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("s_") {
                if let Some(s_array) = self.s.get(name) {
                    let shape = s_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.s.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// RAdam optimizer (Rectified Adam) with variance warmup (ICLR 2020).
///
/// RAdam addresses the bad convergence problem of Adam in the early stages
/// by rectifying the variance of the adaptive learning rate. It provides
/// a variance warmup mechanism that stabilizes training.
///
/// Reference: Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)
#[derive(Debug)]
pub struct RAdamOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl RAdamOptimizer {
    /// Create a new RAdam optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }

    /// Compute the variance rectification term.
    fn compute_rectification(&self) -> (bool, f64) {
        let beta2 = self.config.beta2;
        let t = self.t as f64;

        // Maximum length of the approximated SMA (Simple Moving Average)
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;

        // Length of the approximated SMA at timestep t
        let rho_t = rho_inf - 2.0 * t * beta2.powf(t) / (1.0 - beta2.powf(t));

        // Check if variance is tractable
        if rho_t > 5.0 {
            // Compute rectification term
            let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t);
            (true, rect.sqrt())
        } else {
            // Variance is not tractable, use un-adapted update (like SGD with momentum)
            (false, 0.0)
        }
    }
}

impl Optimizer for RAdamOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);

        // Compute variance rectification
        let (use_adaptive, rect) = self.compute_rectification();

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize moments if not present
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));

            // Bias-corrected first moment
            let m_hat = &*m / bias_correction1;

            if use_adaptive {
                // Use adaptive learning rate with rectification
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                let v_hat = &*v / bias_correction2;

                // Update with rectified variance
                let update = m_hat / (v_hat.mapv(|val| val.sqrt()) + eps);
                *param = &*param - &(update * (lr * rect));
            } else {
                // Early phase: use non-adaptive update (SGD with momentum)
                *param = &*param - &(m_hat * lr);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_val) = state.get("t") {
            self.t = t_val[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m_array) = self.m.get(name) {
                    let shape = m_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v_array) = self.v.get(name) {
                    let shape = v_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// LARS optimizer (Layer-wise Adaptive Rate Scaling).
///
/// LARS scales the learning rate for each layer based on the ratio of the parameter norm
/// to the gradient norm. This is particularly effective for large batch training.
///
/// Reference: You et al. "Large Batch Training of Convolutional Networks" (2017)
#[derive(Debug)]
pub struct LarsOptimizer {
    config: OptimizerConfig,
    /// Momentum buffers for each parameter.
    velocity: HashMap<String, Array<f64, Ix2>>,
    /// Trust coefficient for layer-wise LR adaptation (typically 0.001).
    trust_coef: f64,
    /// Whether to apply LARS to bias parameters.
    exclude_bias: bool,
}

impl LarsOptimizer {
    /// Create a new LARS optimizer.
    ///
    /// # Arguments
    /// * `config` - Optimizer configuration
    /// * `trust_coef` - Trust coefficient for adaptive LR (default: 0.001)
    /// * `exclude_bias` - Whether to exclude bias from LARS adaptation (default: true)
    pub fn new(config: OptimizerConfig, trust_coef: f64, exclude_bias: bool) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
            trust_coef,
            exclude_bias,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }

    /// Compute layer-wise adaptive learning rate.
    fn compute_adaptive_lr(
        &self,
        param: &Array<f64, Ix2>,
        grad: &Array<f64, Ix2>,
        name: &str,
    ) -> f64 {
        // Skip LARS for bias if configured
        if self.exclude_bias && (name.contains("bias") || name.contains("b")) {
            return self.config.learning_rate;
        }

        // Compute parameter norm
        let param_norm: f64 = param.iter().map(|&p| p * p).sum::<f64>().sqrt();

        // Compute gradient norm
        let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();

        // Avoid division by zero
        if param_norm == 0.0 || grad_norm == 0.0 {
            return self.config.learning_rate;
        }

        // Compute layer-wise LR: trust_coef * ||param|| / ||grad||
        let local_lr = self.trust_coef * param_norm / grad_norm;

        // Return base LR * local LR
        self.config.learning_rate * local_lr
    }
}

impl Optimizer for LarsOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Compute layer-wise adaptive learning rate (before borrowing velocity)
            let adaptive_lr = self.compute_adaptive_lr(param, grad, name);

            // Weight decay
            let mut effective_grad = grad.clone();
            if self.config.weight_decay > 0.0 {
                effective_grad += &(&*param * self.config.weight_decay);
            }

            // Initialize velocity if not present
            if !self.velocity.contains_key(name) {
                self.velocity
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let velocity = self.velocity.get_mut(name).unwrap();

            // Update velocity with LARS-adapted LR: v = momentum * v + adaptive_lr * grad
            velocity.mapv_inplace(|v| self.config.momentum * v);
            *velocity = &*velocity + &(effective_grad * adaptive_lr);

            // Update parameter: param = param - velocity
            *param = &*param - &*velocity;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("trust_coef".to_string(), vec![self.trust_coef]);
        state.insert(
            "exclude_bias".to_string(),
            vec![if self.exclude_bias { 1.0 } else { 0.0 }],
        );

        for (name, velocity) in &self.velocity {
            state.insert(
                format!("velocity_{}", name),
                velocity.iter().copied().collect(),
            );
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(trust) = state.get("trust_coef") {
            self.trust_coef = trust[0];
        }
        if let Some(exclude) = state.get("exclude_bias") {
            self.exclude_bias = exclude[0] > 0.5;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("velocity_") {
                if let Some(velocity) = self.velocity.get(name) {
                    let shape = velocity.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.velocity.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

/// SAM optimizer (Sharpness Aware Minimization).
///
/// SAM seeks parameters that lie in neighborhoods having uniformly low loss,
/// improving model generalization. It requires two forward-backward passes per step:
/// one to compute the adversarial perturbation, and one to compute the actual gradient.
///
/// Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
///
/// Note: This is a wrapper optimizer. SAM requires special handling in the training loop
/// to perform two gradient computations per step. The typical usage is:
/// 1. Compute gradients at current parameters
/// 2. Compute adversarial perturbation
/// 3. Compute gradients at perturbed parameters
/// 4. Update with the perturbed gradients
#[derive(Debug)]
pub struct SamOptimizer<O: Optimizer> {
    /// Base optimizer (e.g., SGD, Adam).
    base_optimizer: O,
    /// Perturbation radius (rho).
    rho: f64,
    /// Stored perturbations for each parameter.
    perturbations: HashMap<String, Array<f64, Ix2>>,
}

impl<O: Optimizer> SamOptimizer<O> {
    /// Create a new SAM optimizer.
    ///
    /// # Arguments
    /// * `base_optimizer` - The base optimizer to use (SGD, Adam, etc.)
    /// * `rho` - Perturbation radius (typically 0.05)
    pub fn new(base_optimizer: O, rho: f64) -> TrainResult<Self> {
        if rho <= 0.0 {
            return Err(TrainError::OptimizerError(
                "SAM rho must be positive".to_string(),
            ));
        }

        Ok(Self {
            base_optimizer,
            rho,
            perturbations: HashMap::new(),
        })
    }

    /// Compute adversarial perturbations.
    ///
    /// This should be called with the first set of gradients to compute
    /// the perturbation direction.
    pub fn first_step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        // Compute gradient norm
        let grad_norm = compute_gradient_norm(gradients);

        if grad_norm == 0.0 {
            return Ok(());
        }

        // Compute and apply perturbations: e = rho * grad / ||grad||
        for (name, param) in parameters.iter_mut() {
            let grad = gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Compute perturbation
            let perturbation = grad.mapv(|g| self.rho * g / grad_norm);

            // Apply perturbation: param = param + e
            *param = &*param + &perturbation;

            // Store perturbation for later removal
            self.perturbations.insert(name.clone(), perturbation);
        }

        Ok(())
    }

    /// Perform the actual optimization step.
    ///
    /// This should be called with the second set of gradients (computed at the perturbed parameters).
    /// It will remove the perturbations and update the parameters using the base optimizer.
    pub fn second_step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        // Remove perturbations first: param = param - e
        for (name, param) in parameters.iter_mut() {
            if let Some(perturbation) = self.perturbations.get(name) {
                *param = &*param - perturbation;
            }
        }

        // Clear perturbations
        self.perturbations.clear();

        // Perform base optimizer step with the gradients at perturbed point
        self.base_optimizer.step(parameters, gradients)
    }
}

impl<O: Optimizer> Optimizer for SamOptimizer<O> {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        // For the trait implementation, we just do the second step
        // In practice, users should call first_step() and second_step() explicitly
        self.second_step(parameters, gradients)
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.base_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = self.base_optimizer.state_dict();
        state.insert("rho".to_string(), vec![self.rho]);

        for (name, perturbation) in &self.perturbations {
            state.insert(
                format!("perturbation_{}", name),
                perturbation.iter().copied().collect(),
            );
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(rho_val) = state.get("rho") {
            self.rho = rho_val[0];
        }

        // Load base optimizer state
        self.base_optimizer.load_state_dict(state.clone());

        // Load perturbations
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("perturbation_") {
                if let Some(pert) = self.perturbations.get(name) {
                    let shape = pert.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.perturbations.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sgd_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
    }

    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let mut optimizer = AdamOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_adamw_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdamWOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            grad_clip: Some(0.05),
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[10.0]]); // Large gradient

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        // With clipping, gradient should be capped at 0.05
        assert!((w[[0, 0]] - (1.0 - 0.1 * 0.05)).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut optimizer = RMSpropOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_adagrad_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = AdagradOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
    }

    #[test]
    fn test_nadam_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.002,
            ..Default::default()
        };
        let mut optimizer = NAdamOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_lamb_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = LambOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_adamax_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.002,
            ..Default::default()
        };
        let mut optimizer = AdaMaxOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);

        // Perform multiple steps to test infinity norm tracking
        for _ in 0..3 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let w = params.get("w").unwrap();
        // Parameters should decrease
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
        assert!(w[[1, 0]] < 3.0);
        assert!(w[[1, 1]] < 4.0);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("u_w"));
    }

    #[test]
    fn test_lookahead_optimizer() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let inner_optimizer = AdamOptimizer::new(inner_config);

        let mut optimizer = LookaheadOptimizer::new(inner_optimizer, 0.5, 5).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);

        // Step several times
        for _ in 0..10 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let w = params.get("w").unwrap();
        // Parameters should decrease
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);

        // Test learning rate access
        assert_eq!(optimizer.get_lr(), 0.01);

        optimizer.set_lr(0.02);
        assert_eq!(optimizer.get_lr(), 0.02);
    }

    #[test]
    fn test_lookahead_invalid_alpha() {
        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());

        let result = LookaheadOptimizer::new(inner_optimizer, 1.5, 5);
        assert!(result.is_err());

        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());
        let result = LookaheadOptimizer::new(inner_optimizer, -0.1, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_lookahead_invalid_k() {
        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());

        let result = LookaheadOptimizer::new(inner_optimizer, 0.5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lookahead_synchronization() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let inner_optimizer = SgdOptimizer::new(inner_config);

        let mut optimizer = LookaheadOptimizer::new(inner_optimizer, 0.5, 3).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1]]);

        let initial_w = params.get("w").unwrap()[[0, 0]];

        // Step 3 times to trigger synchronization
        for _ in 0..3 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let w_after_sync = params.get("w").unwrap()[[0, 0]];

        // Parameters should have changed after synchronization
        assert_ne!(w_after_sync, initial_w);
        assert!(w_after_sync < initial_w);
    }

    #[test]
    fn test_adabelief_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdaBeliefOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);

        // Perform multiple steps
        for _ in 0..5 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let w = params.get("w").unwrap();
        // Parameters should decrease
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[1, 1]] < 4.0);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("s_w"));
    }

    #[test]
    fn test_radam_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let mut optimizer = RAdamOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        // Perform multiple steps (RAdam needs warmup)
        for _ in 0..10 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let w = params.get("w").unwrap();
        // Parameters should decrease
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("v_w"));
    }

    #[test]
    fn test_lars_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0001,
            ..Default::default()
        };
        let mut optimizer = LarsOptimizer::new(config, 0.001, true);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        // Parameters should decrease
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[1, 1]] < 4.0);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("trust_coef"));
        assert!(state.contains_key("exclude_bias"));
        assert!(state.contains_key("velocity_w"));
    }

    #[test]
    fn test_lars_bias_exclusion() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            ..Default::default()
        };

        // Test with exclude_bias = true
        let mut optimizer = LarsOptimizer::new(config.clone(), 0.001, true);

        let mut params = HashMap::new();
        params.insert("weights".to_string(), array![[1.0, 2.0]]);
        params.insert("bias".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("weights".to_string(), array![[0.1, 0.1]]);
        grads.insert("bias".to_string(), array![[0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        // Both should decrease, but bias should use base LR
        let weights = params.get("weights").unwrap();
        let bias = params.get("bias").unwrap();
        assert!(weights[[0, 0]] < 1.0);
        assert!(bias[[0, 0]] < 1.0);
    }

    #[test]
    fn test_sam_optimizer() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let inner_optimizer = SgdOptimizer::new(inner_config);

        let mut optimizer = SamOptimizer::new(inner_optimizer, 0.05).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);

        // First step: compute perturbation
        let original_w = params.get("w").unwrap().clone();
        optimizer.first_step(&mut params, &grads).unwrap();

        // Parameters should be perturbed
        let perturbed_w = params.get("w").unwrap();
        assert_ne!(perturbed_w[[0, 0]], original_w[[0, 0]]);

        // Second step: update with gradients at perturbed point
        optimizer.second_step(&mut params, &grads).unwrap();

        // Parameters should be updated from original position
        let final_w = params.get("w").unwrap();
        assert!(final_w[[0, 0]] < original_w[[0, 0]]);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("rho"));
    }

    #[test]
    fn test_sam_invalid_rho() {
        let inner_optimizer = SgdOptimizer::new(OptimizerConfig::default());

        let result = SamOptimizer::new(inner_optimizer, 0.0);
        assert!(result.is_err());

        let inner_optimizer = SgdOptimizer::new(OptimizerConfig::default());
        let result = SamOptimizer::new(inner_optimizer, -0.1);
        assert!(result.is_err());
    }
}
