//! Learning rate schedulers.

use crate::Optimizer;

/// Trait for learning rate schedulers.
pub trait LrScheduler {
    /// Update learning rate based on current step/epoch.
    fn step(&mut self, optimizer: &mut dyn Optimizer);

    /// Get current learning rate.
    fn get_lr(&self) -> f64;

    /// Get scheduler state as a dictionary.
    fn state_dict(&self) -> std::collections::HashMap<String, f64>;

    /// Load scheduler state from a dictionary.
    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()>;
}

/// Step-based learning rate scheduler.
/// Decreases learning rate by a factor every `step_size` epochs.
#[derive(Debug, Clone)]
pub struct StepLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Step size (epochs).
    pub step_size: usize,
    /// Multiplicative factor of learning rate decay.
    pub gamma: f64,
    /// Current epoch counter.
    current_epoch: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl StepLrScheduler {
    /// Create a new step LR scheduler.
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LrScheduler for StepLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;

        if self.current_epoch.is_multiple_of(self.step_size) {
            self.current_lr *= self.gamma;
            optimizer.set_lr(self.current_lr);
        }
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_epoch".to_string(), self.current_epoch as f64);
        state.insert("step_size".to_string(), self.step_size as f64);
        state.insert("gamma".to_string(), self.gamma);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_epoch) = state.get("current_epoch") {
            self.current_epoch = current_epoch as usize;
        }
        Ok(())
    }
}

/// Exponential learning rate scheduler.
/// Decreases learning rate by a factor of gamma every epoch.
#[derive(Debug, Clone)]
pub struct ExponentialLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Multiplicative factor of learning rate decay.
    pub gamma: f64,
    /// Current epoch counter.
    current_epoch: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl ExponentialLrScheduler {
    /// Create a new exponential LR scheduler.
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LrScheduler for ExponentialLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;
        self.current_lr = self.initial_lr * self.gamma.powi(self.current_epoch as i32);
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_epoch".to_string(), self.current_epoch as f64);
        state.insert("gamma".to_string(), self.gamma);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_epoch) = state.get("current_epoch") {
            self.current_epoch = current_epoch as usize;
        }
        Ok(())
    }
}

/// Cosine annealing learning rate scheduler.
/// Anneals learning rate using a cosine schedule.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Minimum learning rate.
    pub min_lr: f64,
    /// Total number of epochs.
    pub t_max: usize,
    /// Current epoch counter.
    current_epoch: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl CosineAnnealingLrScheduler {
    /// Create a new cosine annealing LR scheduler.
    pub fn new(initial_lr: f64, min_lr: f64, t_max: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LrScheduler for CosineAnnealingLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;

        let progress = (self.current_epoch as f64) / (self.t_max as f64);
        let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay;

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_epoch".to_string(), self.current_epoch as f64);
        state.insert("min_lr".to_string(), self.min_lr);
        state.insert("t_max".to_string(), self.t_max as f64);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_epoch) = state.get("current_epoch") {
            self.current_epoch = current_epoch as usize;
        }
        Ok(())
    }
}

/// Warmup scheduler that linearly increases learning rate.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WarmupScheduler {
    /// Target learning rate after warmup.
    pub target_lr: f64,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl WarmupScheduler {
    /// Create a new warmup scheduler.
    #[allow(dead_code)]
    pub fn new(target_lr: f64, warmup_steps: usize) -> Self {
        Self {
            target_lr,
            warmup_steps,
            current_step: 0,
            current_lr: 0.0,
        }
    }
}

impl LrScheduler for WarmupScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;

        if self.current_step < self.warmup_steps {
            self.current_lr =
                self.target_lr * (self.current_step as f64) / (self.warmup_steps as f64);
        } else {
            self.current_lr = self.target_lr;
        }

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("target_lr".to_string(), self.target_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        Ok(())
    }
}

/// One-cycle learning rate scheduler.
/// Increases LR from initial to max, then decreases to min.
#[derive(Debug, Clone)]
pub struct OneCycleLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Maximum learning rate.
    pub max_lr: f64,
    /// Minimum learning rate (final).
    pub min_lr: f64,
    /// Total number of steps.
    pub total_steps: usize,
    /// Percentage of cycle spent increasing LR.
    pub pct_start: f64,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl OneCycleLrScheduler {
    /// Create a new one-cycle LR scheduler.
    pub fn new(
        initial_lr: f64,
        max_lr: f64,
        min_lr: f64,
        total_steps: usize,
        pct_start: f64,
    ) -> Self {
        Self {
            initial_lr,
            max_lr,
            min_lr,
            total_steps,
            pct_start,
            current_step: 0,
            current_lr: initial_lr,
        }
    }
}

impl LrScheduler for OneCycleLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;

        let step_num = self.current_step.min(self.total_steps);
        let pct = step_num as f64 / self.total_steps as f64;

        if pct < self.pct_start {
            // Increasing phase
            let phase_pct = pct / self.pct_start;
            self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * phase_pct;
        } else {
            // Decreasing phase
            let phase_pct = (pct - self.pct_start) / (1.0 - self.pct_start);
            // Cosine annealing for smooth decay
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * phase_pct).cos());
            self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay;
        }

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("max_lr".to_string(), self.max_lr);
        state.insert("min_lr".to_string(), self.min_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("total_steps".to_string(), self.total_steps as f64);
        state.insert("pct_start".to_string(), self.pct_start);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        Ok(())
    }
}

/// Polynomial decay learning rate scheduler.
#[derive(Debug, Clone)]
pub struct PolynomialDecayLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Final learning rate.
    pub final_lr: f64,
    /// Power of the polynomial.
    pub power: f64,
    /// Total number of decay steps.
    pub decay_steps: usize,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl PolynomialDecayLrScheduler {
    /// Create a new polynomial decay LR scheduler.
    pub fn new(initial_lr: f64, final_lr: f64, power: f64, decay_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            power,
            decay_steps,
            current_step: 0,
            current_lr: initial_lr,
        }
    }
}

impl LrScheduler for PolynomialDecayLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;

        let step_num = self.current_step.min(self.decay_steps);
        let decay_factor = (1.0 - (step_num as f64 / self.decay_steps as f64)).powf(self.power);
        self.current_lr = (self.initial_lr - self.final_lr) * decay_factor + self.final_lr;

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("final_lr".to_string(), self.final_lr);
        state.insert("power".to_string(), self.power);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("decay_steps".to_string(), self.decay_steps as f64);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        Ok(())
    }
}

/// Cyclic learning rate mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CyclicLrMode {
    /// Triangular (linear increase and decrease).
    Triangular,
    /// Triangular2 (amplitude decreases by half each cycle).
    Triangular2,
    /// Exponential range (amplitude decreases exponentially).
    ExpRange,
}

/// Cyclic learning rate scheduler.
#[derive(Debug, Clone)]
pub struct CyclicLrScheduler {
    /// Base learning rate.
    pub base_lr: f64,
    /// Maximum learning rate.
    pub max_lr: f64,
    /// Step size (half of cycle length).
    pub step_size: usize,
    /// Cyclic mode.
    pub mode: CyclicLrMode,
    /// Gamma for exponential range mode.
    pub gamma: f64,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
    /// Current cycle number.
    cycle: usize,
}

impl CyclicLrScheduler {
    /// Create a new cyclic LR scheduler.
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize, mode: CyclicLrMode) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size,
            mode,
            gamma: 0.99994,
            current_step: 0,
            current_lr: base_lr,
            cycle: 0,
        }
    }

    /// Create a new cyclic LR scheduler with exponential range mode.
    pub fn new_exp_range(base_lr: f64, max_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size,
            mode: CyclicLrMode::ExpRange,
            gamma,
            current_step: 0,
            current_lr: base_lr,
            cycle: 0,
        }
    }
}

impl LrScheduler for CyclicLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;

        // Determine cycle position
        let cycle = (self.current_step - 1) / (2 * self.step_size);
        let x = ((self.current_step - 1) as f64 / self.step_size as f64).abs() % 2.0;

        // Calculate scaling factor based on mode
        let scale_fn = match self.mode {
            CyclicLrMode::Triangular => 1.0,
            CyclicLrMode::Triangular2 => 1.0 / 2.0_f64.powi(cycle as i32),
            CyclicLrMode::ExpRange => self.gamma.powi(self.current_step as i32),
        };

        // Calculate current LR
        if x <= 1.0 {
            // Increasing phase
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * x * scale_fn;
        } else {
            // Decreasing phase
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * (2.0 - x) * scale_fn;
        }

        self.cycle = cycle;
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("base_lr".to_string(), self.base_lr);
        state.insert("max_lr".to_string(), self.max_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("step_size".to_string(), self.step_size as f64);
        state.insert("cycle".to_string(), self.cycle as f64);
        state.insert("gamma".to_string(), self.gamma);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        if let Some(&cycle) = state.get("cycle") {
            self.cycle = cycle as usize;
        }
        Ok(())
    }
}

/// Warmup with cosine annealing scheduler.
#[derive(Debug, Clone)]
pub struct WarmupCosineLrScheduler {
    /// Target learning rate after warmup.
    pub target_lr: f64,
    /// Minimum learning rate.
    pub min_lr: f64,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Total number of steps (including warmup).
    pub total_steps: usize,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl WarmupCosineLrScheduler {
    /// Create a new warmup cosine LR scheduler.
    pub fn new(target_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            target_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            current_lr: 0.0,
        }
    }
}

impl LrScheduler for WarmupCosineLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;

        if self.current_step <= self.warmup_steps {
            // Warmup phase: linear increase
            self.current_lr =
                self.target_lr * (self.current_step as f64 / self.warmup_steps as f64);
        } else {
            // Cosine annealing phase
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.current_lr = self.min_lr + (self.target_lr - self.min_lr) * cosine_decay;
        }

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("target_lr".to_string(), self.target_lr);
        state.insert("min_lr".to_string(), self.min_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        state.insert("total_steps".to_string(), self.total_steps as f64);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        Ok(())
    }
}

/// Noam scheduler (Transformer learning rate schedule).
///
/// This is the learning rate schedule used in "Attention is All You Need".
/// It increases linearly for warmup_steps, then decays proportionally to the
/// inverse square root of the step number.
///
/// Reference: Vaswani et al. "Attention is All You Need" (NIPS 2017)
#[derive(Debug, Clone)]
pub struct NoamScheduler {
    /// Model dimension (d_model) from the paper.
    model_dim: f64,
    /// Number of warmup steps.
    warmup_steps: usize,
    /// Scaling factor (typically 1.0).
    scale_factor: f64,
    /// Current step counter.
    current_step: usize,
    /// Current learning rate.
    current_lr: f64,
}

impl NoamScheduler {
    /// Create a new Noam scheduler.
    ///
    /// # Arguments
    /// * `model_dim` - Model dimension (d_model), typically 512 for Transformer
    /// * `warmup_steps` - Number of warmup steps, typically 4000
    /// * `scale_factor` - Scaling factor, typically 1.0 or 2.0
    pub fn new(model_dim: usize, warmup_steps: usize, scale_factor: f64) -> Self {
        let model_dim_f64 = model_dim as f64;
        let current_lr = scale_factor * model_dim_f64.powf(-0.5);

        Self {
            model_dim: model_dim_f64,
            warmup_steps,
            scale_factor,
            current_step: 0,
            current_lr,
        }
    }

    /// Compute learning rate for the current step.
    fn compute_lr(&self) -> f64 {
        let step = (self.current_step + 1) as f64; // +1 to avoid division by zero
        let warmup = self.warmup_steps as f64;

        // lr = scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        self.scale_factor
            * self.model_dim.powf(-0.5)
            * step.powf(-0.5).min(step * warmup.powf(-1.5))
    }
}

impl LrScheduler for NoamScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        self.current_lr = self.compute_lr();
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("model_dim".to_string(), self.model_dim);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        state.insert("scale_factor".to_string(), self.scale_factor);
        state.insert("current_step".to_string(), self.current_step as f64);
        state.insert("current_lr".to_string(), self.current_lr);
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_step) = state.get("current_step") {
            self.current_step = current_step as usize;
        }
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        Ok(())
    }
}

/// Multi-step learning rate scheduler.
///
/// Decays the learning rate by gamma at specified milestones (epochs).
/// This is useful when you know specific points where you want to reduce LR.
#[derive(Debug, Clone)]
pub struct MultiStepLrScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Milestones (epochs) at which to decay LR.
    pub milestones: Vec<usize>,
    /// Multiplicative factor of learning rate decay.
    pub gamma: f64,
    /// Current epoch counter.
    current_epoch: usize,
    /// Current learning rate.
    current_lr: f64,
    /// Index of next milestone to trigger.
    next_milestone_idx: usize,
}

impl MultiStepLrScheduler {
    /// Create a new multi-step LR scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `milestones` - Epochs at which to decay (should be sorted)
    /// * `gamma` - Multiplicative decay factor
    pub fn new(initial_lr: f64, mut milestones: Vec<usize>, gamma: f64) -> Self {
        // Ensure milestones are sorted
        milestones.sort_unstable();

        Self {
            initial_lr,
            milestones,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
            next_milestone_idx: 0,
        }
    }
}

impl LrScheduler for MultiStepLrScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;

        // Check if we've reached a milestone
        if self.next_milestone_idx < self.milestones.len()
            && self.current_epoch >= self.milestones[self.next_milestone_idx]
        {
            self.current_lr *= self.gamma;
            self.next_milestone_idx += 1;
            optimizer.set_lr(self.current_lr);
        }
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("initial_lr".to_string(), self.initial_lr);
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("current_epoch".to_string(), self.current_epoch as f64);
        state.insert("gamma".to_string(), self.gamma);
        state.insert(
            "next_milestone_idx".to_string(),
            self.next_milestone_idx as f64,
        );
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&current_epoch) = state.get("current_epoch") {
            self.current_epoch = current_epoch as usize;
        }
        if let Some(&next_milestone_idx) = state.get("next_milestone_idx") {
            self.next_milestone_idx = next_milestone_idx as usize;
        }
        Ok(())
    }
}

/// Reduce learning rate on plateau (metric-based adaptive scheduler).
///
/// Reduces learning rate when a metric (e.g., validation loss) has stopped improving.
/// This scheduler requires explicit metric updates via `step_with_metric()`.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateauScheduler {
    /// Current learning rate.
    current_lr: f64,
    /// Decay factor.
    pub factor: f64,
    /// Number of epochs with no improvement after which LR will be reduced.
    pub patience: usize,
    /// Minimum LR.
    pub min_lr: f64,
    /// Threshold for measuring improvement (relative).
    pub threshold: f64,
    /// Number of epochs to wait before resuming normal operation after LR reduction.
    pub cooldown: usize,
    /// Best metric value seen so far.
    best_metric: Option<f64>,
    /// Number of epochs with no improvement.
    num_bad_epochs: usize,
    /// Epochs remaining in cooldown period.
    cooldown_counter: usize,
    /// Mode: "min" (lower is better) or "max" (higher is better).
    mode: PlateauMode,
}

/// Mode for ReduceLROnPlateau scheduler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlateauMode {
    /// Lower metric values are better (e.g., loss).
    Min,
    /// Higher metric values are better (e.g., accuracy).
    Max,
}

impl ReduceLROnPlateauScheduler {
    /// Create a new ReduceLROnPlateau scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `mode` - Whether to minimize or maximize the metric
    /// * `factor` - Factor by which to reduce LR (new_lr = lr * factor)
    /// * `patience` - Number of epochs with no improvement to wait
    /// * `threshold` - Threshold for measuring improvement (relative)
    /// * `min_lr` - Minimum LR (won't reduce below this)
    /// * `cooldown` - Cooldown epochs after LR reduction
    pub fn new(
        initial_lr: f64,
        mode: PlateauMode,
        factor: f64,
        patience: usize,
        threshold: f64,
        min_lr: f64,
        cooldown: usize,
    ) -> Self {
        Self {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            threshold,
            cooldown,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            mode,
        }
    }

    /// Step with a metric value.
    ///
    /// This should be called with the validation metric at the end of each epoch.
    pub fn step_with_metric(&mut self, optimizer: &mut dyn Optimizer, metric: f64) {
        // Check if in cooldown period
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return;
        }

        // Check if metric has improved
        let is_better = match self.best_metric {
            None => true, // First metric always sets the baseline
            Some(best) => match self.mode {
                PlateauMode::Min => metric < best * (1.0 - self.threshold),
                PlateauMode::Max => metric > best * (1.0 + self.threshold),
            },
        };

        if is_better {
            // Metric improved
            self.best_metric = Some(metric);
            self.num_bad_epochs = 0;
        } else {
            // Metric didn't improve
            self.num_bad_epochs += 1;

            if self.num_bad_epochs >= self.patience {
                // Reduce learning rate
                let new_lr = (self.current_lr * self.factor).max(self.min_lr);

                if new_lr < self.current_lr {
                    self.current_lr = new_lr;
                    optimizer.set_lr(self.current_lr);
                    self.cooldown_counter = self.cooldown;
                    self.num_bad_epochs = 0;
                }
            }
        }
    }
}

impl LrScheduler for ReduceLROnPlateauScheduler {
    fn step(&mut self, _optimizer: &mut dyn Optimizer) {
        // This scheduler needs metrics, so the default step() does nothing.
        // Users should call step_with_metric() instead.
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f64> {
        let mut state = std::collections::HashMap::new();
        state.insert("current_lr".to_string(), self.current_lr);
        state.insert("factor".to_string(), self.factor);
        state.insert("patience".to_string(), self.patience as f64);
        state.insert("min_lr".to_string(), self.min_lr);
        state.insert("threshold".to_string(), self.threshold);
        state.insert("cooldown".to_string(), self.cooldown as f64);
        state.insert(
            "best_metric".to_string(),
            self.best_metric.unwrap_or(f64::NAN),
        );
        state.insert("num_bad_epochs".to_string(), self.num_bad_epochs as f64);
        state.insert("cooldown_counter".to_string(), self.cooldown_counter as f64);
        state.insert(
            "mode".to_string(),
            match self.mode {
                PlateauMode::Min => 0.0,
                PlateauMode::Max => 1.0,
            },
        );
        state
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, f64>,
    ) -> crate::TrainResult<()> {
        if let Some(&current_lr) = state.get("current_lr") {
            self.current_lr = current_lr;
        }
        if let Some(&best_metric) = state.get("best_metric") {
            self.best_metric = if best_metric.is_nan() {
                None
            } else {
                Some(best_metric)
            };
        }
        if let Some(&num_bad_epochs) = state.get("num_bad_epochs") {
            self.num_bad_epochs = num_bad_epochs as usize;
        }
        if let Some(&cooldown_counter) = state.get("cooldown_counter") {
            self.cooldown_counter = cooldown_counter as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OptimizerConfig, SgdOptimizer};

    #[test]
    fn test_step_lr_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = StepLrScheduler::new(0.1, 2, 0.5);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.get_lr(), 0.05);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.get_lr(), 0.05);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.get_lr(), 0.025);
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = ExponentialLrScheduler::new(0.1, 0.9);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.09).abs() < 1e-6);

        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = CosineAnnealingLrScheduler::new(0.1, 0.01, 10);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert!(scheduler.get_lr() < 0.1);
        assert!(scheduler.get_lr() > 0.01);

        // Step to halfway point
        for _ in 1..5 {
            scheduler.step(&mut optimizer);
        }
        let halfway_lr = scheduler.get_lr();
        assert!((halfway_lr - 0.055).abs() < 0.01); // Should be approximately at midpoint
    }

    #[test]
    fn test_warmup_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.0,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = WarmupScheduler::new(0.1, 10);

        assert_eq!(scheduler.get_lr(), 0.0);

        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);

        for _ in 1..10 {
            scheduler.step(&mut optimizer);
        }
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.get_lr(), 0.1); // Stays at target after warmup
    }

    #[test]
    fn test_one_cycle_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = OneCycleLrScheduler::new(0.01, 0.1, 0.001, 100, 0.3);

        assert_eq!(scheduler.get_lr(), 0.01);

        // Test increasing phase
        for _ in 0..30 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() > 0.01);
        assert!(scheduler.get_lr() <= 0.1);

        // Test decreasing phase
        for _ in 30..100 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() < 0.1);
    }

    #[test]
    fn test_polynomial_decay_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = PolynomialDecayLrScheduler::new(0.1, 0.001, 2.0, 100);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut optimizer);
        assert!(scheduler.get_lr() < 0.1);

        for _ in 1..100 {
            scheduler.step(&mut optimizer);
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_cyclic_lr_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = CyclicLrScheduler::new(0.01, 0.1, 10, CyclicLrMode::Triangular);

        assert_eq!(scheduler.get_lr(), 0.01);

        // Test first cycle
        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() > 0.01);

        for _ in 10..20 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() < 0.1);
    }

    #[test]
    fn test_warmup_cosine_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.0,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = WarmupCosineLrScheduler::new(0.1, 0.001, 10, 100);

        assert_eq!(scheduler.get_lr(), 0.0);

        // Test warmup phase
        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // Test middle of cosine annealing phase
        for _ in 10..50 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() < 0.1);
        assert!(scheduler.get_lr() > 0.001);

        // Test near end of cosine annealing phase
        for _ in 50..100 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() < 0.1);
        // At the end, LR should be close to min_lr
        assert!((scheduler.get_lr() - 0.001).abs() < 0.01);
    }

    #[test]
    fn test_noam_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.0,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = NoamScheduler::new(512, 4000, 1.0);

        let initial_lr = scheduler.get_lr();
        assert!(initial_lr > 0.0);

        // Step once
        scheduler.step(&mut optimizer);
        let step1_lr = scheduler.get_lr();

        // After step, LR should change
        assert!(step1_lr != initial_lr);

        // At peak (warmup_steps), test decrease after that
        for _ in 1..4000 {
            scheduler.step(&mut optimizer);
        }
        let peak_lr = scheduler.get_lr();

        // After warmup, LR should decrease
        for _ in 4000..8000 {
            scheduler.step(&mut optimizer);
        }
        assert!(scheduler.get_lr() < peak_lr);
    }

    #[test]
    fn test_multistep_lr_scheduler() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = MultiStepLrScheduler::new(0.1, vec![10, 20, 30], 0.1);

        assert_eq!(scheduler.get_lr(), 0.1);

        // Before first milestone
        for _ in 0..9 {
            scheduler.step(&mut optimizer);
        }
        assert_eq!(scheduler.get_lr(), 0.1);

        // At first milestone (epoch 10)
        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);

        // Between first and second milestone
        for _ in 10..19 {
            scheduler.step(&mut optimizer);
        }
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);

        // At second milestone (epoch 20)
        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);

        // At third milestone (epoch 30)
        for _ in 20..29 {
            scheduler.step(&mut optimizer);
        }
        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_mode() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = ReduceLROnPlateauScheduler::new(
            0.1,              // initial_lr
            PlateauMode::Min, // mode
            0.5,              // factor
            3,                // patience
            0.01,             // threshold
            0.001,            // min_lr
            2,                // cooldown
        );

        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric improving - LR should not change
        scheduler.step_with_metric(&mut optimizer, 1.0);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric plateaus for patience epochs
        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step_with_metric(&mut optimizer, 0.9);
        // After patience epochs, LR should be reduced
        assert_eq!(scheduler.get_lr(), 0.05);

        // Test cooldown - LR shouldn't change during cooldown
        scheduler.step_with_metric(&mut optimizer, 1.0);
        assert_eq!(scheduler.get_lr(), 0.05);

        scheduler.step_with_metric(&mut optimizer, 1.0);
        assert_eq!(scheduler.get_lr(), 0.05);
    }

    #[test]
    fn test_reduce_lr_on_plateau_max_mode() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);
        let mut scheduler = ReduceLROnPlateauScheduler::new(
            0.1,
            PlateauMode::Max, // Maximize metric (e.g., accuracy)
            0.1,
            2,
            0.01,
            0.001,
            0,
        );

        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric improving (increasing) - LR should not change
        scheduler.step_with_metric(&mut optimizer, 0.5);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step_with_metric(&mut optimizer, 0.6);
        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric plateaus
        scheduler.step_with_metric(&mut optimizer, 0.6);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step_with_metric(&mut optimizer, 0.6);
        // After patience epochs, LR should be reduced
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
    }
}
