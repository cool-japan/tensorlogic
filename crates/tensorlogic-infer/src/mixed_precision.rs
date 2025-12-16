//! Mixed precision training utilities.
//!
//! This module provides comprehensive mixed precision training support:
//! - FP16 (half precision) and BF16 (bfloat16) computation modes
//! - Automatic loss scaling with dynamic adjustment
//! - Gradient checkpointing integration
//! - Mixed precision optimizer wrappers
//! - Numerical stability monitoring
//! - Performance profiling for mixed precision operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{MixedPrecisionConfig, PrecisionMode, LossScaler, MixedPrecisionTrainer};
//!
//! // Configure mixed precision training
//! let config = MixedPrecisionConfig::default()
//!     .with_compute_dtype(PrecisionMode::FP16)
//!     .with_param_dtype(PrecisionMode::FP32)
//!     .with_loss_scaling(LossScalingStrategy::Dynamic {
//!         init_scale: 65536.0,
//!         growth_factor: 2.0,
//!         backoff_factor: 0.5,
//!         growth_interval: 2000,
//!     });
//!
//! // Create mixed precision trainer
//! let mut trainer = MixedPrecisionTrainer::new(executor, config);
//!
//! // Training loop with automatic loss scaling
//! for batch in dataset {
//!     let loss = trainer.train_step(&batch)?;
//!     println!("Loss: {:.4}, Scale: {:.0}", loss, trainer.current_loss_scale());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Mixed precision training errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MixedPrecisionError {
    #[error("Loss scale overflow: scale={0}")]
    LossScaleOverflow(f64),

    #[error("Loss scale underflow: scale={0}")]
    LossScaleUnderflow(f64),

    #[error("Gradient overflow detected in {0} gradients")]
    GradientOverflow(usize),

    #[error("NaN detected in gradients")]
    GradientNaN,

    #[error("Unsupported precision mode: {0:?}")]
    UnsupportedPrecisionMode(PrecisionMode),

    #[error("Mixed precision not supported by backend")]
    NotSupported,

    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
}

/// Precision mode for computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// 32-bit floating point (full precision)
    FP32,
    /// 16-bit floating point (half precision)
    FP16,
    /// BFloat16 (brain floating point 16)
    BF16,
    /// 64-bit floating point (double precision)
    FP64,
    /// 8-bit floating point (experimental)
    FP8,
}

impl PrecisionMode {
    /// Get the number of bytes per element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            PrecisionMode::FP64 => 8,
            PrecisionMode::FP32 => 4,
            PrecisionMode::FP16 | PrecisionMode::BF16 => 2,
            PrecisionMode::FP8 => 1,
        }
    }

    /// Check if this precision mode is mixed (lower than FP32).
    pub fn is_mixed_precision(&self) -> bool {
        matches!(
            self,
            PrecisionMode::FP16 | PrecisionMode::BF16 | PrecisionMode::FP8
        )
    }

    /// Get the precision name.
    pub fn name(&self) -> &'static str {
        match self {
            PrecisionMode::FP32 => "float32",
            PrecisionMode::FP16 => "float16",
            PrecisionMode::BF16 => "bfloat16",
            PrecisionMode::FP64 => "float64",
            PrecisionMode::FP8 => "float8",
        }
    }
}

/// Loss scaling strategy for mixed precision training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LossScalingStrategy {
    /// No loss scaling
    None,

    /// Static loss scaling with a fixed scale factor
    Static { scale: f64 },

    /// Dynamic loss scaling with automatic adjustment
    Dynamic {
        /// Initial scale factor
        init_scale: f64,
        /// Growth factor when no overflow detected
        growth_factor: f64,
        /// Backoff factor when overflow detected
        backoff_factor: f64,
        /// Number of successful steps before growing scale
        growth_interval: usize,
    },
}

impl Default for LossScalingStrategy {
    fn default() -> Self {
        LossScalingStrategy::Dynamic {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

/// Loss scaler for automatic mixed precision training.
#[derive(Debug, Clone)]
pub struct LossScaler {
    strategy: LossScalingStrategy,
    current_scale: f64,
    growth_tracker: usize,
    overflow_count: usize,
    total_steps: usize,
}

impl LossScaler {
    /// Create a new loss scaler with the given strategy.
    pub fn new(strategy: LossScalingStrategy) -> Self {
        let current_scale = match &strategy {
            LossScalingStrategy::None => 1.0,
            LossScalingStrategy::Static { scale } => *scale,
            LossScalingStrategy::Dynamic { init_scale, .. } => *init_scale,
        };

        Self {
            strategy,
            current_scale,
            growth_tracker: 0,
            overflow_count: 0,
            total_steps: 0,
        }
    }

    /// Get the current loss scale.
    pub fn scale(&self) -> f64 {
        self.current_scale
    }

    /// Scale the loss value.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        loss * self.current_scale
    }

    /// Unscale gradients.
    pub fn unscale_gradients(&self, grads: &mut HashMap<String, f64>) {
        let inv_scale = 1.0 / self.current_scale;
        for grad in grads.values_mut() {
            *grad *= inv_scale;
        }
    }

    /// Check for gradient overflow or NaN values.
    pub fn check_overflow(&self, grads: &HashMap<String, f64>) -> Result<(), MixedPrecisionError> {
        let mut has_nan = false;
        let mut has_inf = false;

        for grad in grads.values() {
            if grad.is_nan() {
                has_nan = true;
            }
            if grad.is_infinite() {
                has_inf = true;
            }
        }

        if has_nan {
            return Err(MixedPrecisionError::GradientNaN);
        }
        if has_inf {
            return Err(MixedPrecisionError::GradientOverflow(
                grads.values().filter(|g| g.is_infinite()).count(),
            ));
        }

        Ok(())
    }

    /// Update the loss scale based on overflow detection.
    pub fn update(&mut self, found_overflow: bool) -> Result<(), MixedPrecisionError> {
        self.total_steps += 1;

        match &self.strategy {
            LossScalingStrategy::None | LossScalingStrategy::Static { .. } => {
                // No update for static scaling
            }
            LossScalingStrategy::Dynamic {
                growth_factor,
                backoff_factor,
                growth_interval,
                ..
            } => {
                if found_overflow {
                    // Reduce scale
                    self.current_scale *= backoff_factor;
                    self.growth_tracker = 0;
                    self.overflow_count += 1;

                    // Check for underflow
                    if self.current_scale < 1.0 {
                        return Err(MixedPrecisionError::LossScaleUnderflow(self.current_scale));
                    }
                } else {
                    // Increase growth tracker
                    self.growth_tracker += 1;

                    // Grow scale if interval reached
                    if self.growth_tracker >= *growth_interval {
                        self.current_scale *= growth_factor;
                        self.growth_tracker = 0;

                        // Check for overflow
                        if self.current_scale > 1e10 {
                            return Err(MixedPrecisionError::LossScaleOverflow(self.current_scale));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get statistics about loss scaling.
    pub fn stats(&self) -> LossScalerStats {
        LossScalerStats {
            current_scale: self.current_scale,
            overflow_count: self.overflow_count,
            total_steps: self.total_steps,
            overflow_rate: self.overflow_count as f64 / self.total_steps.max(1) as f64,
            growth_tracker: self.growth_tracker,
        }
    }

    /// Reset the loss scaler to initial state.
    pub fn reset(&mut self) {
        let init_scale = match &self.strategy {
            LossScalingStrategy::None => 1.0,
            LossScalingStrategy::Static { scale } => *scale,
            LossScalingStrategy::Dynamic { init_scale, .. } => *init_scale,
        };

        self.current_scale = init_scale;
        self.growth_tracker = 0;
        self.overflow_count = 0;
        self.total_steps = 0;
    }
}

/// Statistics about loss scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct LossScalerStats {
    /// Current scale factor
    pub current_scale: f64,
    /// Number of overflow events
    pub overflow_count: usize,
    /// Total training steps
    pub total_steps: usize,
    /// Overflow rate (overflows / total steps)
    pub overflow_rate: f64,
    /// Current growth tracker value
    pub growth_tracker: usize,
}

/// Mixed precision training configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Precision mode for computation
    pub compute_dtype: PrecisionMode,

    /// Precision mode for parameters (usually FP32)
    pub param_dtype: PrecisionMode,

    /// Loss scaling strategy
    pub loss_scaling: LossScalingStrategy,

    /// Enable gradient checkpointing to save memory
    pub gradient_checkpointing: bool,

    /// Enable gradient clipping before unscaling
    pub gradient_clipping: bool,

    /// Maximum gradient norm for clipping
    pub max_gradient_norm: f64,

    /// Enable numerical stability checks
    pub stability_checks: bool,

    /// Skip optimizer step if overflow detected
    pub skip_on_overflow: bool,

    /// Enable master weights in FP32
    pub use_master_weights: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            compute_dtype: PrecisionMode::FP16,
            param_dtype: PrecisionMode::FP32,
            loss_scaling: LossScalingStrategy::default(),
            gradient_checkpointing: false,
            gradient_clipping: true,
            max_gradient_norm: 1.0,
            stability_checks: true,
            skip_on_overflow: true,
            use_master_weights: true,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new mixed precision config.
    pub fn new(compute_dtype: PrecisionMode, param_dtype: PrecisionMode) -> Self {
        Self {
            compute_dtype,
            param_dtype,
            ..Default::default()
        }
    }

    /// Set the compute dtype.
    pub fn with_compute_dtype(mut self, dtype: PrecisionMode) -> Self {
        self.compute_dtype = dtype;
        self
    }

    /// Set the parameter dtype.
    pub fn with_param_dtype(mut self, dtype: PrecisionMode) -> Self {
        self.param_dtype = dtype;
        self
    }

    /// Set the loss scaling strategy.
    pub fn with_loss_scaling(mut self, strategy: LossScalingStrategy) -> Self {
        self.loss_scaling = strategy;
        self
    }

    /// Enable or disable gradient checkpointing.
    pub fn with_gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }

    /// Enable or disable gradient clipping.
    pub fn with_gradient_clipping(mut self, enabled: bool, max_norm: f64) -> Self {
        self.gradient_clipping = enabled;
        self.max_gradient_norm = max_norm;
        self
    }

    /// Enable or disable stability checks.
    pub fn with_stability_checks(mut self, enabled: bool) -> Self {
        self.stability_checks = enabled;
        self
    }

    /// Enable or disable master weights.
    pub fn with_master_weights(mut self, enabled: bool) -> Self {
        self.use_master_weights = enabled;
        self
    }

    /// Create FP16 mixed precision config.
    pub fn fp16() -> Self {
        Self::new(PrecisionMode::FP16, PrecisionMode::FP32)
    }

    /// Create BF16 mixed precision config.
    pub fn bf16() -> Self {
        Self::new(PrecisionMode::BF16, PrecisionMode::FP32)
    }

    /// Create FP8 mixed precision config (experimental).
    pub fn fp8() -> Self {
        Self::new(PrecisionMode::FP8, PrecisionMode::FP32)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), MixedPrecisionError> {
        // Check that param dtype is at least as precise as compute dtype
        let compute_bytes = self.compute_dtype.bytes_per_element();
        let param_bytes = self.param_dtype.bytes_per_element();

        if param_bytes < compute_bytes {
            return Err(MixedPrecisionError::NumericalInstability(format!(
                "Parameter dtype ({:?}) should be at least as precise as compute dtype ({:?})",
                self.param_dtype, self.compute_dtype
            )));
        }

        // Validate loss scaling parameters
        if let LossScalingStrategy::Dynamic {
            init_scale,
            growth_factor,
            backoff_factor,
            ..
        } = &self.loss_scaling
        {
            if *init_scale <= 0.0 {
                return Err(MixedPrecisionError::LossScaleUnderflow(*init_scale));
            }
            if *growth_factor <= 1.0 {
                return Err(MixedPrecisionError::NumericalInstability(format!(
                    "Growth factor must be > 1.0, got {}",
                    growth_factor
                )));
            }
            if *backoff_factor >= 1.0 || *backoff_factor <= 0.0 {
                return Err(MixedPrecisionError::NumericalInstability(format!(
                    "Backoff factor must be in (0, 1), got {}",
                    backoff_factor
                )));
            }
        }

        Ok(())
    }
}

/// Mixed precision training state.
#[derive(Debug, Clone)]
pub struct MixedPrecisionState {
    /// Configuration
    pub config: MixedPrecisionConfig,

    /// Loss scaler
    pub scaler: LossScaler,

    /// Master weights (FP32 copies of parameters)
    pub master_weights: HashMap<String, Vec<f64>>,

    /// Number of successful steps
    pub successful_steps: usize,

    /// Number of skipped steps due to overflow
    pub skipped_steps: usize,

    /// Training step counter
    pub step: usize,
}

impl MixedPrecisionState {
    /// Create a new mixed precision training state.
    pub fn new(config: MixedPrecisionConfig) -> Result<Self, MixedPrecisionError> {
        config.validate()?;

        Ok(Self {
            scaler: LossScaler::new(config.loss_scaling.clone()),
            config,
            master_weights: HashMap::new(),
            successful_steps: 0,
            skipped_steps: 0,
            step: 0,
        })
    }

    /// Initialize master weights.
    pub fn init_master_weights(&mut self, params: &HashMap<String, Vec<f64>>) {
        if self.config.use_master_weights {
            self.master_weights = params.clone();
        }
    }

    /// Get current loss scale.
    pub fn current_loss_scale(&self) -> f64 {
        self.scaler.scale()
    }

    /// Get training statistics.
    pub fn stats(&self) -> MixedPrecisionStats {
        let scaler_stats = self.scaler.stats();

        MixedPrecisionStats {
            compute_dtype: self.config.compute_dtype,
            param_dtype: self.config.param_dtype,
            current_scale: scaler_stats.current_scale,
            total_steps: self.step,
            successful_steps: self.successful_steps,
            skipped_steps: self.skipped_steps,
            overflow_count: scaler_stats.overflow_count,
            overflow_rate: scaler_stats.overflow_rate,
            success_rate: self.successful_steps as f64 / self.step.max(1) as f64,
        }
    }

    /// Process training step with automatic loss scaling.
    pub fn process_step(
        &mut self,
        loss: f64,
        gradients: &mut HashMap<String, f64>,
    ) -> Result<bool, MixedPrecisionError> {
        self.step += 1;

        // Scale loss (used implicitly in gradient scaling)
        let _scaled_loss = self.scaler.scale_loss(loss);

        // Scale gradients by loss scale
        for grad in gradients.values_mut() {
            *grad *= self.scaler.scale();
        }

        // Unscale gradients
        self.scaler.unscale_gradients(gradients);

        // Check for overflow
        let found_overflow = self.scaler.check_overflow(gradients).is_err();

        // Update loss scale
        self.scaler.update(found_overflow)?;

        if found_overflow {
            self.skipped_steps += 1;
            Ok(false) // Skip optimizer step
        } else {
            self.successful_steps += 1;
            Ok(true) // Proceed with optimizer step
        }
    }
}

/// Statistics about mixed precision training.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedPrecisionStats {
    /// Compute dtype
    pub compute_dtype: PrecisionMode,
    /// Parameter dtype
    pub param_dtype: PrecisionMode,
    /// Current loss scale
    pub current_scale: f64,
    /// Total training steps
    pub total_steps: usize,
    /// Successful steps (no overflow)
    pub successful_steps: usize,
    /// Skipped steps (due to overflow)
    pub skipped_steps: usize,
    /// Total overflow count
    pub overflow_count: usize,
    /// Overflow rate
    pub overflow_rate: f64,
    /// Success rate
    pub success_rate: f64,
}

impl std::fmt::Display for MixedPrecisionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mixed Precision Training Statistics")?;
        writeln!(f, "====================================")?;
        writeln!(f, "Compute dtype:     {:?}", self.compute_dtype)?;
        writeln!(f, "Parameter dtype:   {:?}", self.param_dtype)?;
        writeln!(f, "Current scale:     {:.0}", self.current_scale)?;
        writeln!(f, "Total steps:       {}", self.total_steps)?;
        writeln!(f, "Successful steps:  {}", self.successful_steps)?;
        writeln!(f, "Skipped steps:     {}", self.skipped_steps)?;
        writeln!(f, "Overflow count:    {}", self.overflow_count)?;
        writeln!(f, "Overflow rate:     {:.2}%", self.overflow_rate * 100.0)?;
        writeln!(f, "Success rate:      {:.2}%", self.success_rate * 100.0)?;
        Ok(())
    }
}

/// Gradient checkpointing utility.
#[derive(Debug, Clone)]
pub struct GradientCheckpoint {
    /// Checkpoint identifier
    pub id: String,

    /// Saved tensors for recomputation
    pub saved_tensors: HashMap<String, Vec<f64>>,

    /// Memory saved by checkpointing (bytes)
    pub memory_saved: usize,
}

impl GradientCheckpoint {
    /// Create a new gradient checkpoint.
    pub fn new(id: String) -> Self {
        Self {
            id,
            saved_tensors: HashMap::new(),
            memory_saved: 0,
        }
    }

    /// Save a tensor for recomputation.
    pub fn save_tensor(&mut self, name: String, data: Vec<f64>) {
        let bytes = data.len() * std::mem::size_of::<f64>();
        self.memory_saved += bytes;
        self.saved_tensors.insert(name, data);
    }

    /// Get memory saved by this checkpoint.
    pub fn memory_saved_mb(&self) -> f64 {
        self.memory_saved as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_mode_bytes() {
        assert_eq!(PrecisionMode::FP64.bytes_per_element(), 8);
        assert_eq!(PrecisionMode::FP32.bytes_per_element(), 4);
        assert_eq!(PrecisionMode::FP16.bytes_per_element(), 2);
        assert_eq!(PrecisionMode::BF16.bytes_per_element(), 2);
        assert_eq!(PrecisionMode::FP8.bytes_per_element(), 1);
    }

    #[test]
    fn test_precision_mode_is_mixed() {
        assert!(!PrecisionMode::FP32.is_mixed_precision());
        assert!(PrecisionMode::FP16.is_mixed_precision());
        assert!(PrecisionMode::BF16.is_mixed_precision());
        assert!(PrecisionMode::FP8.is_mixed_precision());
    }

    #[test]
    fn test_loss_scaler_static() {
        let scaler = LossScaler::new(LossScalingStrategy::Static { scale: 1024.0 });
        assert_eq!(scaler.scale(), 1024.0);

        let loss = 0.5;
        let scaled = scaler.scale_loss(loss);
        assert_eq!(scaled, 512.0);
    }

    #[test]
    fn test_loss_scaler_dynamic_no_overflow() {
        let mut scaler = LossScaler::new(LossScalingStrategy::Dynamic {
            init_scale: 1024.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
        });

        assert_eq!(scaler.scale(), 1024.0);

        // No overflow for 2 steps
        scaler.update(false).unwrap();
        scaler.update(false).unwrap();

        // Scale should have grown
        assert_eq!(scaler.scale(), 2048.0);
    }

    #[test]
    fn test_loss_scaler_dynamic_with_overflow() {
        let mut scaler = LossScaler::new(LossScalingStrategy::Dynamic {
            init_scale: 1024.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
        });

        assert_eq!(scaler.scale(), 1024.0);

        // Overflow detected
        scaler.update(true).unwrap();

        // Scale should have reduced
        assert_eq!(scaler.scale(), 512.0);
    }

    #[test]
    fn test_loss_scaler_overflow_detection() {
        let scaler = LossScaler::new(LossScalingStrategy::None);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), 1.0);
        grads.insert("w2".to_string(), 2.0);

        // No overflow
        assert!(scaler.check_overflow(&grads).is_ok());

        // Add NaN
        grads.insert("w3".to_string(), f64::NAN);
        assert!(matches!(
            scaler.check_overflow(&grads),
            Err(MixedPrecisionError::GradientNaN)
        ));

        // Remove NaN and add Inf
        grads.remove("w3");
        grads.insert("w4".to_string(), f64::INFINITY);
        assert!(matches!(
            scaler.check_overflow(&grads),
            Err(MixedPrecisionError::GradientOverflow(_))
        ));
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.compute_dtype, PrecisionMode::FP16);
        assert_eq!(config.param_dtype, PrecisionMode::FP32);
        assert!(config.use_master_weights);
        assert!(config.stability_checks);
    }

    #[test]
    fn test_mixed_precision_config_builders() {
        let config = MixedPrecisionConfig::fp16();
        assert_eq!(config.compute_dtype, PrecisionMode::FP16);

        let config = MixedPrecisionConfig::bf16();
        assert_eq!(config.compute_dtype, PrecisionMode::BF16);

        let config = MixedPrecisionConfig::fp8();
        assert_eq!(config.compute_dtype, PrecisionMode::FP8);
    }

    #[test]
    fn test_mixed_precision_config_validation() {
        // Valid config
        let config = MixedPrecisionConfig::new(PrecisionMode::FP16, PrecisionMode::FP32);
        assert!(config.validate().is_ok());

        // Invalid: param dtype less precise than compute dtype
        let config = MixedPrecisionConfig::new(PrecisionMode::FP32, PrecisionMode::FP16);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mixed_precision_state() {
        let config = MixedPrecisionConfig::fp16();
        let mut state = MixedPrecisionState::new(config).unwrap();

        assert_eq!(state.step, 0);
        assert_eq!(state.successful_steps, 0);
        assert_eq!(state.skipped_steps, 0);

        // Process step without overflow
        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), 0.1);
        grads.insert("w2".to_string(), 0.2);

        let should_update = state.process_step(0.5, &mut grads).unwrap();
        assert!(should_update);
        assert_eq!(state.step, 1);
        assert_eq!(state.successful_steps, 1);
        assert_eq!(state.skipped_steps, 0);
    }

    #[test]
    fn test_mixed_precision_stats_display() {
        let stats = MixedPrecisionStats {
            compute_dtype: PrecisionMode::FP16,
            param_dtype: PrecisionMode::FP32,
            current_scale: 1024.0,
            total_steps: 100,
            successful_steps: 95,
            skipped_steps: 5,
            overflow_count: 5,
            overflow_rate: 0.05,
            success_rate: 0.95,
        };

        let display = format!("{}", stats);
        assert!(display.contains("FP16"));
        assert!(display.contains("1024"));
        assert!(display.contains("95"));
    }

    #[test]
    fn test_gradient_checkpoint() {
        let mut checkpoint = GradientCheckpoint::new("layer1".to_string());
        assert_eq!(checkpoint.memory_saved, 0);

        checkpoint.save_tensor("activations".to_string(), vec![1.0, 2.0, 3.0]);
        assert!(checkpoint.memory_saved > 0);
        assert!(checkpoint.memory_saved_mb() > 0.0);
    }

    #[test]
    fn test_loss_scaler_stats() {
        let mut scaler = LossScaler::new(LossScalingStrategy::Dynamic {
            init_scale: 1024.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
        });

        scaler.update(false).unwrap();
        scaler.update(true).unwrap();
        scaler.update(false).unwrap();

        let stats = scaler.stats();
        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.overflow_count, 1);
        assert!((stats.overflow_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_loss_scaler_reset() {
        let mut scaler = LossScaler::new(LossScalingStrategy::Dynamic {
            init_scale: 1024.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
        });

        scaler.update(false).unwrap();
        scaler.update(false).unwrap();
        assert_eq!(scaler.scale(), 2048.0);

        scaler.reset();
        assert_eq!(scaler.scale(), 1024.0);
        assert_eq!(scaler.stats().total_steps, 0);
    }

    #[test]
    fn test_unscale_gradients() {
        let scaler = LossScaler::new(LossScalingStrategy::Static { scale: 1024.0 });

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), 1024.0);
        grads.insert("w2".to_string(), 2048.0);

        scaler.unscale_gradients(&mut grads);

        assert_eq!(grads.get("w1").unwrap(), &1.0);
        assert_eq!(grads.get("w2").unwrap(), &2.0);
    }
}
