//! Mixed precision training infrastructure for memory efficiency and speed.
//!
//! This module provides utilities for training with reduced precision (FP16/BF16)
//! while maintaining numerical stability through loss scaling and gradient management.
//!
//! # Features
//! - FP16 and BF16 precision modes
//! - Dynamic and static loss scaling
//! - Gradient overflow detection
//! - Master weight management
//! - Automatic precision casting
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_train::{MixedPrecisionTrainer, PrecisionMode, LossScaler};
//!
//! // Create FP16 trainer with dynamic loss scaling
//! let mut trainer = MixedPrecisionTrainer::new(
//!     PrecisionMode::FP16,
//!     LossScaler::dynamic(2.0_f32.powi(15), 2.0, 2000),
//! );
//!
//! // Train with automatic precision management
//! trainer.scale_loss(loss);
//! ```

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::TrainResult;

/// Precision mode for mixed precision training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Full precision (FP32) - baseline
    FP32,
    /// Half precision (FP16) - 2x memory reduction
    FP16,
    /// Brain floating point (BF16) - better for training
    BF16,
}

impl PrecisionMode {
    /// Returns the number of bytes per element for this precision.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            PrecisionMode::FP32 => 4,
            PrecisionMode::FP16 => 2,
            PrecisionMode::BF16 => 2,
        }
    }

    /// Returns the memory reduction factor compared to FP32.
    pub fn memory_reduction(&self) -> f32 {
        match self {
            PrecisionMode::FP32 => 1.0,
            PrecisionMode::FP16 => 2.0,
            PrecisionMode::BF16 => 2.0,
        }
    }

    /// Returns the typical numerical range for this precision.
    pub fn numerical_range(&self) -> (f32, f32) {
        match self {
            PrecisionMode::FP32 => (-3.4e38, 3.4e38),
            PrecisionMode::FP16 => (-6.55e4, 6.55e4),
            PrecisionMode::BF16 => (-3.39e38, 3.39e38), // Same exponent range as FP32
        }
    }
}

/// Loss scaling strategy for mixed precision training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossScaler {
    /// No loss scaling (not recommended for FP16)
    None,
    /// Static loss scaling with fixed scale factor
    Static { scale: f32 },
    /// Dynamic loss scaling that adjusts based on gradient overflow
    Dynamic {
        /// Current scale factor
        scale: f32,
        /// Growth factor when no overflow (typically 2.0)
        growth_factor: f32,
        /// Backoff factor when overflow detected (typically 0.5)
        backoff_factor: f32,
        /// Number of successful steps before growing scale
        growth_interval: usize,
        /// Current step counter
        steps_since_overflow: usize,
    },
}

impl LossScaler {
    /// Creates a static loss scaler.
    pub fn static_scale(scale: f32) -> Self {
        Self::Static { scale }
    }

    /// Creates a dynamic loss scaler with typical defaults.
    ///
    /// # Arguments
    /// * `initial_scale` - Starting scale (typically 2^15 = 32768)
    /// * `growth_factor` - How much to grow scale (typically 2.0)
    /// * `growth_interval` - Steps before growing (typically 2000)
    pub fn dynamic(initial_scale: f32, growth_factor: f32, growth_interval: usize) -> Self {
        Self::Dynamic {
            scale: initial_scale,
            growth_factor,
            backoff_factor: 0.5,
            growth_interval,
            steps_since_overflow: 0,
        }
    }

    /// Gets the current scale factor.
    pub fn get_scale(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Static { scale } => *scale,
            Self::Dynamic { scale, .. } => *scale,
        }
    }

    /// Scales a loss value.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.get_scale()
    }

    /// Unscales gradients (divides by scale).
    pub fn unscale_gradients(&self, gradients: &mut Array2<f32>) {
        let scale = self.get_scale();
        if scale != 1.0 {
            *gradients /= scale;
        }
    }

    /// Updates the dynamic scaler based on overflow detection.
    ///
    /// # Arguments
    /// * `overflow_detected` - Whether gradient overflow was detected
    ///
    /// # Returns
    /// True if the optimizer step should proceed
    pub fn update(&mut self, overflow_detected: bool) -> bool {
        if let Self::Dynamic {
            scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_overflow,
        } = self
        {
            if overflow_detected {
                // Backoff: reduce scale and reset counter
                *scale *= *backoff_factor;
                *steps_since_overflow = 0;
                false // Skip optimizer step
            } else {
                // Increment counter
                *steps_since_overflow += 1;

                // Grow scale if interval reached
                if *steps_since_overflow >= *growth_interval {
                    *scale *= *growth_factor;
                    *steps_since_overflow = 0;
                }
                true // Proceed with optimizer step
            }
        } else {
            // Static or None scaling always proceeds
            !overflow_detected
        }
    }
}

/// Mixed precision training manager.
pub struct MixedPrecisionTrainer {
    /// Precision mode
    mode: PrecisionMode,
    /// Loss scaler
    scaler: LossScaler,
    /// Master weights (FP32) - keeps full precision copy
    master_weights: HashMap<String, Array2<f32>>,
    /// Training statistics
    stats: MixedPrecisionStats,
}

impl MixedPrecisionTrainer {
    /// Creates a new mixed precision trainer.
    pub fn new(mode: PrecisionMode, scaler: LossScaler) -> Self {
        Self {
            mode,
            scaler,
            master_weights: HashMap::new(),
            stats: MixedPrecisionStats::default(),
        }
    }

    /// Registers weights to maintain master copy.
    pub fn register_weights(&mut self, name: String, weights: Array2<f32>) {
        self.master_weights.insert(name, weights);
    }

    /// Converts FP32 weights to working precision.
    pub fn cast_to_working_precision(&self, weights: &Array2<f32>) -> Array2<f32> {
        match self.mode {
            PrecisionMode::FP32 => weights.clone(),
            PrecisionMode::FP16 => self.simulate_fp16(weights),
            PrecisionMode::BF16 => self.simulate_bf16(weights),
        }
    }

    /// Simulates FP16 precision (in FP32 container for compatibility).
    fn simulate_fp16(&self, weights: &Array2<f32>) -> Array2<f32> {
        weights.mapv(|x| {
            // Clamp to FP16 range
            let clamped = x.clamp(-65504.0, 65504.0);
            // Simulate reduced mantissa precision (10 bits vs 23 bits)
            let scale = 2.0_f32.powi(10);
            (clamped * scale).round() / scale
        })
    }

    /// Simulates BF16 precision (in FP32 container for compatibility).
    fn simulate_bf16(&self, weights: &Array2<f32>) -> Array2<f32> {
        weights.mapv(|x| {
            // BF16 has same exponent range as FP32, reduced mantissa (7 bits vs 23 bits)
            let scale = 2.0_f32.powi(7);
            (x * scale).round() / scale
        })
    }

    /// Scales loss for backward pass.
    pub fn scale_loss(&mut self, loss: f32) -> f32 {
        self.stats.total_steps += 1;
        self.scaler.scale_loss(loss)
    }

    /// Unscales and checks gradients for overflow.
    ///
    /// # Returns
    /// (should_step, overflow_detected)
    pub fn unscale_and_check_gradients(
        &mut self,
        gradients: &mut HashMap<String, Array2<f32>>,
    ) -> TrainResult<(bool, bool)> {
        // Check for overflow before unscaling
        let mut overflow = false;
        for (_name, grad) in gradients.iter() {
            if grad.iter().any(|&x| !x.is_finite()) {
                overflow = true;
                break;
            }
        }

        if overflow {
            self.stats.overflow_steps += 1;
        }

        // Unscale gradients
        for (_name, grad) in gradients.iter_mut() {
            self.scaler.unscale_gradients(grad);
        }

        // Update scaler and determine if we should step
        let should_step = self.scaler.update(overflow);

        Ok((should_step, overflow))
    }

    /// Updates master weights from working precision weights.
    pub fn update_master_weights(&mut self, updates: &HashMap<String, Array2<f32>>) {
        for (name, update) in updates {
            if let Some(master) = self.master_weights.get_mut(name) {
                *master = master.clone() + update;
            }
        }
    }

    /// Gets the current precision mode.
    pub fn mode(&self) -> PrecisionMode {
        self.mode
    }

    /// Gets the current loss scale.
    pub fn current_scale(&self) -> f32 {
        self.scaler.get_scale()
    }

    /// Gets training statistics.
    pub fn stats(&self) -> &MixedPrecisionStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = MixedPrecisionStats::default();
    }
}

/// Statistics for mixed precision training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    /// Total training steps attempted
    pub total_steps: usize,
    /// Steps with gradient overflow
    pub overflow_steps: usize,
    /// Successful optimizer steps
    pub successful_steps: usize,
}

impl MixedPrecisionStats {
    /// Calculates overflow rate.
    pub fn overflow_rate(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.overflow_steps as f32 / self.total_steps as f32
        }
    }

    /// Calculates success rate.
    pub fn success_rate(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.successful_steps as f32 / self.total_steps as f32
        }
    }
}

/// Gradient scaler for automatic mixed precision.
pub struct GradientScaler {
    scaler: LossScaler,
    enabled: bool,
}

impl GradientScaler {
    /// Creates a new gradient scaler.
    pub fn new(enabled: bool) -> Self {
        let scaler = if enabled {
            LossScaler::dynamic(2.0_f32.powi(15), 2.0, 2000)
        } else {
            LossScaler::None
        };

        Self { scaler, enabled }
    }

    /// Creates a gradient scaler with custom settings.
    pub fn with_scaler(scaler: LossScaler, enabled: bool) -> Self {
        Self { scaler, enabled }
    }

    /// Scales a loss tensor.
    pub fn scale(&self, loss: f32) -> f32 {
        if self.enabled {
            self.scaler.scale_loss(loss)
        } else {
            loss
        }
    }

    /// Unscales gradients.
    pub fn unscale(&self, gradients: &mut Array2<f32>) {
        if self.enabled {
            self.scaler.unscale_gradients(gradients);
        }
    }

    /// Steps with overflow check.
    pub fn step(&mut self, overflow_detected: bool) -> bool {
        if self.enabled {
            self.scaler.update(overflow_detected)
        } else {
            !overflow_detected
        }
    }

    /// Gets the current scale.
    pub fn get_scale(&self) -> f32 {
        self.scaler.get_scale()
    }
}

/// Automatic Mixed Precision (AMP) context manager.
pub struct AutocastContext {
    enabled: bool,
    mode: PrecisionMode,
}

impl AutocastContext {
    /// Creates a new autocast context.
    pub fn new(enabled: bool, mode: PrecisionMode) -> Self {
        Self { enabled, mode }
    }

    /// Checks if autocast is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Gets the target precision mode.
    pub fn mode(&self) -> PrecisionMode {
        self.mode
    }

    /// Casts tensor to working precision if enabled.
    pub fn cast(&self, tensor: &Array2<f32>) -> Array2<f32> {
        if !self.enabled || self.mode == PrecisionMode::FP32 {
            return tensor.clone();
        }

        match self.mode {
            PrecisionMode::FP16 => self.simulate_fp16(tensor),
            PrecisionMode::BF16 => self.simulate_bf16(tensor),
            PrecisionMode::FP32 => tensor.clone(),
        }
    }

    fn simulate_fp16(&self, tensor: &Array2<f32>) -> Array2<f32> {
        tensor.mapv(|x| {
            let clamped = x.clamp(-65504.0, 65504.0);
            let scale = 2.0_f32.powi(10);
            (clamped * scale).round() / scale
        })
    }

    fn simulate_bf16(&self, tensor: &Array2<f32>) -> Array2<f32> {
        tensor.mapv(|x| {
            let scale = 2.0_f32.powi(7);
            (x * scale).round() / scale
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_precision_mode_properties() {
        assert_eq!(PrecisionMode::FP32.bytes_per_element(), 4);
        assert_eq!(PrecisionMode::FP16.bytes_per_element(), 2);
        assert_eq!(PrecisionMode::BF16.bytes_per_element(), 2);

        assert_eq!(PrecisionMode::FP16.memory_reduction(), 2.0);
        assert_eq!(PrecisionMode::BF16.memory_reduction(), 2.0);
    }

    #[test]
    fn test_static_loss_scaler() {
        let scaler = LossScaler::static_scale(1024.0);
        assert_eq!(scaler.get_scale(), 1024.0);

        let loss = 0.5;
        let scaled = scaler.scale_loss(loss);
        assert_eq!(scaled, 512.0);
    }

    #[test]
    fn test_dynamic_loss_scaler() {
        let mut scaler = LossScaler::dynamic(1000.0, 2.0, 3);
        assert_eq!(scaler.get_scale(), 1000.0);

        // No overflow, should grow after 3 steps
        assert!(scaler.update(false));
        assert!(scaler.update(false));
        assert!(scaler.update(false));
        assert_eq!(scaler.get_scale(), 2000.0); // Grew

        // Overflow, should backoff
        assert!(!scaler.update(true));
        assert_eq!(scaler.get_scale(), 1000.0); // Backoff
    }

    #[test]
    fn test_gradient_unscaling() {
        let mut gradients =
            Array2::from_shape_vec((2, 2), vec![100.0, 200.0, 300.0, 400.0]).unwrap();
        let scaler = LossScaler::static_scale(10.0);

        scaler.unscale_gradients(&mut gradients);

        assert_eq!(gradients[[0, 0]], 10.0);
        assert_eq!(gradients[[0, 1]], 20.0);
        assert_eq!(gradients[[1, 0]], 30.0);
        assert_eq!(gradients[[1, 1]], 40.0);
    }

    #[test]
    fn test_mixed_precision_trainer() {
        let mut trainer =
            MixedPrecisionTrainer::new(PrecisionMode::FP16, LossScaler::static_scale(100.0));

        let loss = 0.5;
        let scaled_loss = trainer.scale_loss(loss);
        assert_eq!(scaled_loss, 50.0);
        assert_eq!(trainer.stats().total_steps, 1);
    }

    #[test]
    fn test_fp16_simulation() {
        let trainer = MixedPrecisionTrainer::new(PrecisionMode::FP16, LossScaler::None);

        let weights =
            Array2::from_shape_vec((2, 2), vec![1.234_567, 100000.0, -100000.0, 0.0001]).unwrap();
        let fp16_weights = trainer.cast_to_working_precision(&weights);

        // Should be quantized
        assert_ne!(fp16_weights[[0, 0]], 1.234_567); // Reduced precision
        assert!(fp16_weights[[0, 0]] > 1.0 && fp16_weights[[0, 0]] < 2.0);

        // Large values should be clamped to FP16 range
        assert!(fp16_weights[[0, 1]] <= 65504.0);
        assert!(fp16_weights[[1, 0]] >= -65504.0);
    }

    #[test]
    fn test_bf16_simulation() {
        let trainer = MixedPrecisionTrainer::new(PrecisionMode::BF16, LossScaler::None);

        let weights =
            Array2::from_shape_vec((2, 2), vec![1.234_567, 100.5, -50.25, 0.125]).unwrap();
        let bf16_weights = trainer.cast_to_working_precision(&weights);

        // Should have reduced mantissa precision
        assert_ne!(bf16_weights[[0, 0]], 1.234_567);
    }

    #[test]
    fn test_overflow_detection() {
        let mut trainer =
            MixedPrecisionTrainer::new(PrecisionMode::FP16, LossScaler::dynamic(1000.0, 2.0, 100));

        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![f32::INFINITY, 1.0, 2.0, 3.0]).unwrap(),
        );

        let (should_step, overflow) = trainer.unscale_and_check_gradients(&mut gradients).unwrap();

        assert!(!should_step);
        assert!(overflow);
        assert_eq!(trainer.stats().overflow_steps, 1);
    }

    #[test]
    fn test_gradient_scaler() {
        let scaler = GradientScaler::new(true);

        let loss = 1.0;
        let scaled = scaler.scale(loss);
        assert!(scaled > loss); // Should be scaled

        let mut grads = Array2::from_shape_vec((2, 2), vec![1000.0; 4]).unwrap();
        scaler.unscale(&mut grads);
        assert!(grads[[0, 0]] < 1000.0); // Should be unscaled
    }

    #[test]
    fn test_autocast_context() {
        let ctx = AutocastContext::new(true, PrecisionMode::FP16);
        assert!(ctx.is_enabled());
        assert_eq!(ctx.mode(), PrecisionMode::FP16);

        let tensor = Array2::from_shape_vec((2, 2), vec![1.234_567; 4]).unwrap();
        let casted = ctx.cast(&tensor);

        // Should have reduced precision
        assert_ne!(casted[[0, 0]], 1.234_567);
    }

    #[test]
    fn test_autocast_disabled() {
        let ctx = AutocastContext::new(false, PrecisionMode::FP16);
        assert!(!ctx.is_enabled());

        let tensor = Array2::from_shape_vec((2, 2), vec![1.234_567; 4]).unwrap();
        let casted = ctx.cast(&tensor);

        // Should be unchanged
        assert_eq!(casted, tensor);
    }

    #[test]
    fn test_master_weights_update() {
        let mut trainer = MixedPrecisionTrainer::new(PrecisionMode::FP16, LossScaler::None);

        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        trainer.register_weights("layer1".to_string(), weights.clone());

        let mut updates = HashMap::new();
        updates.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![0.1, 0.1, 0.1, 0.1]).unwrap(),
        );

        trainer.update_master_weights(&updates);

        let master = &trainer.master_weights["layer1"];
        assert_relative_eq!(master[[0, 0]], 1.1, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_stats() {
        let stats = MixedPrecisionStats {
            total_steps: 100,
            overflow_steps: 5,
            successful_steps: 95,
        };

        assert_eq!(stats.overflow_rate(), 0.05);
        assert_eq!(stats.success_rate(), 0.95);
    }

    #[test]
    fn test_loss_scaler_growth() {
        let mut scaler = LossScaler::dynamic(1000.0, 2.0, 2);

        // First successful step
        assert!(scaler.update(false));
        assert_eq!(scaler.get_scale(), 1000.0);

        // Second successful step - should trigger growth
        assert!(scaler.update(false));
        assert_eq!(scaler.get_scale(), 2000.0);
    }
}
