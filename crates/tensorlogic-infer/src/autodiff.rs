//! Autodiff enhancements for training and optimization.
//!
//! This module extends the basic TlAutodiff trait with:
//! - Gradient accumulation strategies
//! - Custom gradient functions
//! - Gradient clipping and scaling

use std::collections::HashMap;

use tensorlogic_ir::EinsumGraph;

/// Strategy for accumulating gradients
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientAccumulationStrategy {
    /// Standard accumulation (sum gradients)
    Standard,
    /// Average gradients over accumulation steps
    Average,
    /// Gradient checkpointing to save memory
    Checkpointing,
    /// Mixed precision accumulation
    MixedPrecision,
}

/// Configuration for gradient accumulation
#[derive(Debug, Clone)]
pub struct AccumulationConfig {
    pub strategy: GradientAccumulationStrategy,
    pub accumulation_steps: usize,
    pub clear_after_step: bool,
}

impl AccumulationConfig {
    pub fn new(strategy: GradientAccumulationStrategy, steps: usize) -> Self {
        AccumulationConfig {
            strategy,
            accumulation_steps: steps,
            clear_after_step: true,
        }
    }

    pub fn standard(steps: usize) -> Self {
        Self::new(GradientAccumulationStrategy::Standard, steps)
    }

    pub fn average(steps: usize) -> Self {
        Self::new(GradientAccumulationStrategy::Average, steps)
    }

    pub fn checkpointing(steps: usize) -> Self {
        Self::new(GradientAccumulationStrategy::Checkpointing, steps)
    }

    pub fn mixed_precision(steps: usize) -> Self {
        Self::new(GradientAccumulationStrategy::MixedPrecision, steps)
    }
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self::standard(1)
    }
}

/// Gradient clipping strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClippingStrategy {
    /// No clipping
    None,
    /// Clip by value (element-wise)
    ByValue { min: f64, max: f64 },
    /// Clip by global norm
    ByGlobalNorm { max_norm: f64 },
    /// Clip by layer norm
    ByLayerNorm { max_norm: f64 },
}

/// Gradient scaling configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientScaling {
    pub enabled: bool,
    pub initial_scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: usize,
}

impl GradientScaling {
    pub fn new(initial_scale: f64) -> Self {
        GradientScaling {
            enabled: true,
            initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }

    pub fn disabled() -> Self {
        GradientScaling {
            enabled: false,
            initial_scale: 1.0,
            growth_factor: 1.0,
            backoff_factor: 1.0,
            growth_interval: 0,
        }
    }
}

impl Default for GradientScaling {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Complete gradient configuration
#[derive(Debug, Clone)]
pub struct GradientConfig {
    pub accumulation: AccumulationConfig,
    pub clipping: ClippingStrategy,
    pub scaling: GradientScaling,
}

impl GradientConfig {
    pub fn new() -> Self {
        GradientConfig {
            accumulation: AccumulationConfig::default(),
            clipping: ClippingStrategy::None,
            scaling: GradientScaling::default(),
        }
    }

    pub fn with_accumulation(mut self, config: AccumulationConfig) -> Self {
        self.accumulation = config;
        self
    }

    pub fn with_clipping(mut self, strategy: ClippingStrategy) -> Self {
        self.clipping = strategy;
        self
    }

    pub fn with_scaling(mut self, scaling: GradientScaling) -> Self {
        self.scaling = scaling;
        self
    }
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom backward function for a tensor operation
pub type BackwardFn<T, E> = Box<dyn Fn(&T, &[T]) -> Result<Vec<T>, E>>;

/// Registry for custom gradient functions
pub struct CustomGradientRegistry<T, E> {
    gradients: HashMap<String, BackwardFn<T, E>>,
}

impl<T, E> CustomGradientRegistry<T, E> {
    pub fn new() -> Self {
        CustomGradientRegistry {
            gradients: HashMap::new(),
        }
    }

    /// Register a custom backward function for an operation
    pub fn register<F>(&mut self, operation_name: String, backward_fn: F)
    where
        F: Fn(&T, &[T]) -> Result<Vec<T>, E> + 'static,
    {
        self.gradients.insert(operation_name, Box::new(backward_fn));
    }

    /// Get custom gradient function for an operation
    pub fn get(&self, operation_name: &str) -> Option<&BackwardFn<T, E>> {
        self.gradients.get(operation_name)
    }

    /// Check if custom gradient exists
    pub fn has_custom_gradient(&self, operation_name: &str) -> bool {
        self.gradients.contains_key(operation_name)
    }

    /// Remove custom gradient
    pub fn unregister(&mut self, operation_name: &str) -> bool {
        self.gradients.remove(operation_name).is_some()
    }

    /// Get number of registered gradients
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }
}

impl<T, E> Default for CustomGradientRegistry<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient statistics for monitoring
#[derive(Debug, Clone)]
pub struct GradientStats {
    pub global_norm: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
    pub num_parameters: usize,
    pub num_finite: usize,
    pub num_infinite: usize,
    pub num_nan: usize,
}

impl GradientStats {
    pub fn new() -> Self {
        GradientStats {
            global_norm: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            mean_value: 0.0,
            num_parameters: 0,
            num_finite: 0,
            num_infinite: 0,
            num_nan: 0,
        }
    }

    pub fn has_nan(&self) -> bool {
        self.num_nan > 0
    }

    pub fn has_inf(&self) -> bool {
        self.num_infinite > 0
    }

    pub fn is_healthy(&self) -> bool {
        !self.has_nan() && !self.has_inf()
    }

    pub fn finite_ratio(&self) -> f64 {
        if self.num_parameters == 0 {
            return 0.0;
        }
        (self.num_finite as f64) / (self.num_parameters as f64)
    }
}

impl Default for GradientStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for executors with enhanced autodiff capabilities
pub trait TlEnhancedAutodiff {
    type Tensor;
    type Tape;
    type Error;

    /// Execute forward pass with gradient accumulation
    fn forward_with_accumulation(
        &mut self,
        graph: &EinsumGraph,
        config: &AccumulationConfig,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Execute backward pass with gradient clipping
    fn backward_with_clipping(
        &mut self,
        graph: &EinsumGraph,
        loss: &Self::Tensor,
        strategy: ClippingStrategy,
    ) -> Result<Self::Tape, Self::Error>;

    /// Apply gradient scaling
    fn scale_gradients(
        &mut self,
        gradients: &mut Self::Tape,
        scaling: &GradientScaling,
    ) -> Result<(), Self::Error>;

    /// Compute gradient statistics
    fn gradient_stats(&self, gradients: &Self::Tape) -> Result<GradientStats, Self::Error>;

    /// Register custom gradient function
    fn register_custom_gradient(
        &mut self,
        operation_name: String,
        backward_fn: BackwardFn<Self::Tensor, Self::Error>,
    );

    /// Check if custom gradient exists
    fn has_custom_gradient(&self, operation_name: &str) -> bool;
}

/// Gradient accumulator for managing accumulated gradients
pub struct GradientAccumulator<T> {
    accumulated_gradients: Vec<T>,
    accumulation_count: usize,
    config: AccumulationConfig,
}

impl<T: Clone> GradientAccumulator<T> {
    pub fn new(config: AccumulationConfig) -> Self {
        GradientAccumulator {
            accumulated_gradients: Vec::new(),
            accumulation_count: 0,
            config,
        }
    }

    /// Add gradients to accumulator
    pub fn accumulate(&mut self, gradients: Vec<T>) {
        if self.accumulated_gradients.is_empty() {
            self.accumulated_gradients = gradients;
        } else {
            // In real implementation, would add tensors element-wise
            self.accumulated_gradients = gradients;
        }
        self.accumulation_count += 1;
    }

    /// Check if ready to step (accumulated enough)
    pub fn is_ready(&self) -> bool {
        self.accumulation_count >= self.config.accumulation_steps
    }

    /// Get accumulated gradients and optionally reset
    pub fn step(&mut self) -> Vec<T> {
        let gradients = self.accumulated_gradients.clone();

        if self.config.clear_after_step {
            self.clear();
        }

        gradients
    }

    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.accumulated_gradients.clear();
        self.accumulation_count = 0;
    }

    /// Get current accumulation count
    pub fn count(&self) -> usize {
        self.accumulation_count
    }

    pub fn config(&self) -> &AccumulationConfig {
        &self.config
    }
}

/// Gradient clipper for applying clipping strategies
pub struct GradientClipper {
    strategy: ClippingStrategy,
    num_clips: usize,
}

impl GradientClipper {
    pub fn new(strategy: ClippingStrategy) -> Self {
        GradientClipper {
            strategy,
            num_clips: 0,
        }
    }

    /// Check if gradient value should be clipped
    pub fn should_clip(&self, value: f64) -> bool {
        match self.strategy {
            ClippingStrategy::None => false,
            ClippingStrategy::ByValue { min, max } => value < min || value > max,
            ClippingStrategy::ByGlobalNorm { max_norm: _ } => {
                // Would need full gradient to compute global norm
                false
            }
            ClippingStrategy::ByLayerNorm { max_norm: _ } => {
                // Would need layer gradients
                false
            }
        }
    }

    /// Clip a single gradient value
    pub fn clip_value(&mut self, value: f64) -> f64 {
        match self.strategy {
            ClippingStrategy::None => value,
            ClippingStrategy::ByValue { min, max } => {
                if value < min || value > max {
                    self.num_clips += 1;
                }
                value.clamp(min, max)
            }
            ClippingStrategy::ByGlobalNorm { max_norm: _ } => value,
            ClippingStrategy::ByLayerNorm { max_norm: _ } => value,
        }
    }

    /// Get number of clipped values
    pub fn num_clips(&self) -> usize {
        self.num_clips
    }

    /// Reset clip counter
    pub fn reset(&mut self) {
        self.num_clips = 0;
    }

    pub fn strategy(&self) -> ClippingStrategy {
        self.strategy
    }
}

/// Gradient scaler for mixed precision training
pub struct GradientScaler {
    config: GradientScaling,
    current_scale: f64,
    growth_tracker: usize,
}

impl GradientScaler {
    pub fn new(config: GradientScaling) -> Self {
        let current_scale = config.initial_scale;
        GradientScaler {
            config,
            current_scale,
            growth_tracker: 0,
        }
    }

    /// Scale gradients up
    pub fn scale(&self, value: f64) -> f64 {
        if !self.config.enabled {
            return value;
        }
        value * self.current_scale
    }

    /// Unscale gradients (for optimizer step)
    pub fn unscale(&self, value: f64) -> f64 {
        if !self.config.enabled {
            return value;
        }
        value / self.current_scale
    }

    /// Update scale based on gradient health
    pub fn update(&mut self, gradients_healthy: bool) {
        if !self.config.enabled {
            return;
        }

        if gradients_healthy {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.current_scale *= self.config.growth_factor;
                self.growth_tracker = 0;
            }
        } else {
            // Backoff on unhealthy gradients
            self.current_scale *= self.config.backoff_factor;
            self.growth_tracker = 0;
        }
    }

    /// Get current scale factor
    pub fn get_scale(&self) -> f64 {
        self.current_scale
    }

    pub fn config(&self) -> &GradientScaling {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulation_config() {
        let config = AccumulationConfig::standard(4);
        assert_eq!(config.strategy, GradientAccumulationStrategy::Standard);
        assert_eq!(config.accumulation_steps, 4);
        assert!(config.clear_after_step);
    }

    #[test]
    fn test_clipping_strategy() {
        let none = ClippingStrategy::None;
        let by_value = ClippingStrategy::ByValue {
            min: -1.0,
            max: 1.0,
        };
        let by_norm = ClippingStrategy::ByGlobalNorm { max_norm: 1.0 };

        assert_eq!(none, ClippingStrategy::None);
        assert_ne!(by_value, none);
        assert_ne!(by_norm, by_value);
    }

    #[test]
    fn test_gradient_config() {
        let config = GradientConfig::new()
            .with_accumulation(AccumulationConfig::average(4))
            .with_clipping(ClippingStrategy::ByValue {
                min: -1.0,
                max: 1.0,
            });

        assert_eq!(
            config.accumulation.strategy,
            GradientAccumulationStrategy::Average
        );
        assert_eq!(config.accumulation.accumulation_steps, 4);
    }

    #[test]
    fn test_gradient_scaling() {
        let scaling = GradientScaling::new(1024.0);
        assert!(scaling.enabled);
        assert_eq!(scaling.initial_scale, 1024.0);
        assert_eq!(scaling.growth_factor, 2.0);

        let disabled = GradientScaling::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_gradient_stats() {
        let mut stats = GradientStats::new();
        stats.num_parameters = 100;
        stats.num_finite = 95;
        stats.num_nan = 5;
        stats.num_infinite = 0;

        assert!(stats.has_nan());
        assert!(!stats.has_inf());
        assert!(!stats.is_healthy());
        assert_eq!(stats.finite_ratio(), 0.95);
    }

    #[test]
    fn test_custom_gradient_registry() {
        let mut registry: CustomGradientRegistry<f64, String> = CustomGradientRegistry::new();

        registry.register("custom_op".to_string(), |_output, _inputs| {
            Ok(vec![1.0, 2.0, 3.0])
        });

        assert!(registry.has_custom_gradient("custom_op"));
        assert!(!registry.has_custom_gradient("other_op"));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let removed = registry.unregister("custom_op");
        assert!(removed);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_gradient_accumulator() {
        let config = AccumulationConfig::standard(3);
        let mut accumulator: GradientAccumulator<f64> = GradientAccumulator::new(config);

        assert_eq!(accumulator.count(), 0);
        assert!(!accumulator.is_ready());

        accumulator.accumulate(vec![1.0, 2.0, 3.0]);
        assert_eq!(accumulator.count(), 1);
        assert!(!accumulator.is_ready());

        accumulator.accumulate(vec![4.0, 5.0, 6.0]);
        accumulator.accumulate(vec![7.0, 8.0, 9.0]);
        assert!(accumulator.is_ready());

        let _gradients = accumulator.step();
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_gradient_clipper() {
        let mut clipper = GradientClipper::new(ClippingStrategy::ByValue {
            min: -1.0,
            max: 1.0,
        });

        assert!(!clipper.should_clip(0.5));
        assert!(clipper.should_clip(2.0));
        assert!(clipper.should_clip(-2.0));

        let clipped = clipper.clip_value(2.0);
        assert_eq!(clipped, 1.0);
        assert_eq!(clipper.num_clips(), 1);

        let clipped = clipper.clip_value(-2.0);
        assert_eq!(clipped, -1.0);
        assert_eq!(clipper.num_clips(), 2);

        clipper.reset();
        assert_eq!(clipper.num_clips(), 0);
    }

    #[test]
    fn test_gradient_scaler() {
        let config = GradientScaling::new(1024.0);
        let mut scaler = GradientScaler::new(config);

        assert_eq!(scaler.get_scale(), 1024.0);

        let scaled = scaler.scale(2.0);
        assert_eq!(scaled, 2048.0);

        let unscaled = scaler.unscale(2048.0);
        assert_eq!(unscaled, 2.0);

        // Test growth
        scaler.growth_tracker = config.growth_interval - 1;
        scaler.update(true);
        assert_eq!(scaler.get_scale(), 2048.0); // Grew by factor of 2

        // Test backoff
        scaler.update(false);
        assert_eq!(scaler.get_scale(), 1024.0); // Backed off by factor of 0.5
    }

    #[test]
    fn test_gradient_scaler_disabled() {
        let config = GradientScaling::disabled();
        let scaler = GradientScaler::new(config);

        assert_eq!(scaler.scale(2.0), 2.0);
        assert_eq!(scaler.unscale(2.0), 2.0);
    }
}
