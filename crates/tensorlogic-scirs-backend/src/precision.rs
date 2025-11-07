//! Precision control for tensor computations.
//!
//! This module provides abstractions for controlling numerical precision
//! (f32, f64, mixed precision).

use std::fmt;

/// Numerical precision for tensor computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// 32-bit floating point (faster, less memory)
    F32,

    /// 64-bit floating point (more accurate)
    #[default]
    F64,

    /// Mixed precision: f16 for storage, f32 for computation
    Mixed16,

    /// Mixed precision: bf16 for storage, f32 for computation
    BFloat16,
}

impl Precision {
    /// Returns the size in bytes of this precision.
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
            Precision::Mixed16 => 2,  // Storage size
            Precision::BFloat16 => 2, // Storage size
        }
    }

    /// Returns true if this is a mixed precision mode.
    pub fn is_mixed(&self) -> bool {
        matches!(self, Precision::Mixed16 | Precision::BFloat16)
    }

    /// Returns the computation precision (the precision used for actual operations).
    pub fn compute_precision(&self) -> ComputePrecision {
        match self {
            Precision::F32 | Precision::Mixed16 | Precision::BFloat16 => ComputePrecision::F32,
            Precision::F64 => ComputePrecision::F64,
        }
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Precision::F32 => "32-bit floating point",
            Precision::F64 => "64-bit floating point",
            Precision::Mixed16 => "Mixed precision (FP16 storage, FP32 compute)",
            Precision::BFloat16 => "Mixed precision (BF16 storage, FP32 compute)",
        }
    }

    /// Memory savings compared to F64.
    pub fn memory_savings(&self) -> f64 {
        1.0 - (self.size_bytes() as f64 / 8.0)
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::F32 => write!(f, "FP32"),
            Precision::F64 => write!(f, "FP64"),
            Precision::Mixed16 => write!(f, "Mixed-FP16"),
            Precision::BFloat16 => write!(f, "Mixed-BF16"),
        }
    }
}

/// Computation precision (the actual precision used for operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePrecision {
    /// 32-bit computation
    F32,

    /// 64-bit computation
    F64,
}

impl fmt::Display for ComputePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputePrecision::F32 => write!(f, "FP32"),
            ComputePrecision::F64 => write!(f, "FP64"),
        }
    }
}

/// Precision configuration for an executor.
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Default precision for tensors
    pub default_precision: Precision,

    /// Enable automatic mixed precision
    pub auto_mixed_precision: bool,

    /// Loss scaling for mixed precision training
    pub loss_scale: Option<f64>,

    /// Dynamic loss scaling (adjust based on gradients)
    pub dynamic_loss_scaling: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: Precision::F64,
            auto_mixed_precision: false,
            loss_scale: None,
            dynamic_loss_scaling: false,
        }
    }
}

impl PrecisionConfig {
    /// Create a configuration for FP32 precision.
    pub fn f32() -> Self {
        Self {
            default_precision: Precision::F32,
            auto_mixed_precision: false,
            loss_scale: None,
            dynamic_loss_scaling: false,
        }
    }

    /// Create a configuration for FP64 precision.
    pub fn f64() -> Self {
        Self {
            default_precision: Precision::F64,
            auto_mixed_precision: false,
            loss_scale: None,
            dynamic_loss_scaling: false,
        }
    }

    /// Create a configuration for mixed precision training.
    pub fn mixed_precision() -> Self {
        Self {
            default_precision: Precision::Mixed16,
            auto_mixed_precision: true,
            loss_scale: Some(2048.0), // Common starting value
            dynamic_loss_scaling: true,
        }
    }

    /// Enable automatic mixed precision.
    pub fn with_auto_mixed_precision(mut self, enable: bool) -> Self {
        self.auto_mixed_precision = enable;
        self
    }

    /// Set the loss scale for mixed precision training.
    pub fn with_loss_scale(mut self, scale: f64) -> Self {
        self.loss_scale = Some(scale);
        self
    }

    /// Enable dynamic loss scaling.
    pub fn with_dynamic_loss_scaling(mut self, enable: bool) -> Self {
        self.dynamic_loss_scaling = enable;
        self
    }
}

/// Trait for scalar types that can be used in tensor computations.
///
/// This trait abstracts over f32 and f64 for generic tensor operations.
pub trait Scalar:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + std::fmt::Debug
    + std::fmt::Display
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + 'static
{
    /// Zero value
    fn zero() -> Self;

    /// One value
    fn one() -> Self;

    /// Maximum value
    fn max_value() -> Self;

    /// Minimum value (most negative)
    fn min_value() -> Self;

    /// Positive infinity
    fn infinity() -> Self;

    /// Negative infinity
    fn neg_infinity() -> Self;

    /// Not a number
    fn nan() -> Self;

    /// Check if value is NaN
    fn is_nan(self) -> bool;

    /// Check if value is infinite
    fn is_infinite(self) -> bool;

    /// Check if value is finite
    fn is_finite(self) -> bool;

    /// Absolute value
    fn abs(self) -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Exponential
    fn exp(self) -> Self;

    /// Natural logarithm
    fn ln(self) -> Self;

    /// Maximum of two values
    fn max(self, other: Self) -> Self;

    /// Minimum of two values
    fn min(self, other: Self) -> Self;

    /// Convert from f64
    fn from_f64(value: f64) -> Self;

    /// Convert to f64
    fn to_f64(self) -> f64;

    /// The precision type
    fn precision() -> Precision;
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn infinity() -> Self {
        f32::INFINITY
    }

    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }

    fn nan() -> Self {
        f32::NAN
    }

    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn precision() -> Precision {
        Precision::F32
    }
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn infinity() -> Self {
        f64::INFINITY
    }

    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }

    fn nan() -> Self {
        f64::NAN
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn precision() -> Precision {
        Precision::F64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_properties() {
        assert_eq!(Precision::F32.size_bytes(), 4);
        assert_eq!(Precision::F64.size_bytes(), 8);
        assert_eq!(Precision::Mixed16.size_bytes(), 2);

        assert!(!Precision::F32.is_mixed());
        assert!(!Precision::F64.is_mixed());
        assert!(Precision::Mixed16.is_mixed());
    }

    #[test]
    fn test_precision_default() {
        let precision = Precision::default();
        assert_eq!(precision, Precision::F64);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::F32.to_string(), "FP32");
        assert_eq!(Precision::F64.to_string(), "FP64");
        assert_eq!(Precision::Mixed16.to_string(), "Mixed-FP16");
    }

    #[test]
    fn test_precision_memory_savings() {
        assert!((Precision::F32.memory_savings() - 0.5).abs() < 0.01); // 50% savings vs F64
        assert!((Precision::F64.memory_savings()).abs() < 0.01); // 0% savings
        assert!((Precision::Mixed16.memory_savings() - 0.75).abs() < 0.01); // 75% savings
    }

    #[test]
    fn test_precision_config_default() {
        let config = PrecisionConfig::default();
        assert_eq!(config.default_precision, Precision::F64);
        assert!(!config.auto_mixed_precision);
    }

    #[test]
    fn test_precision_config_builders() {
        let f32_config = PrecisionConfig::f32();
        assert_eq!(f32_config.default_precision, Precision::F32);

        let f64_config = PrecisionConfig::f64();
        assert_eq!(f64_config.default_precision, Precision::F64);

        let mixed_config = PrecisionConfig::mixed_precision();
        assert_eq!(mixed_config.default_precision, Precision::Mixed16);
        assert!(mixed_config.auto_mixed_precision);
        assert!(mixed_config.loss_scale.is_some());
    }

    #[test]
    fn test_precision_config_builder_methods() {
        let config = PrecisionConfig::f32()
            .with_auto_mixed_precision(true)
            .with_loss_scale(1024.0)
            .with_dynamic_loss_scaling(true);

        assert!(config.auto_mixed_precision);
        assert_eq!(config.loss_scale, Some(1024.0));
        assert!(config.dynamic_loss_scaling);
    }

    #[test]
    fn test_scalar_f32() {
        assert_eq!(f32::zero(), 0.0_f32);
        assert_eq!(f32::one(), 1.0_f32);
        assert!(f32::infinity().is_infinite());
        assert!(f32::nan().is_nan());

        let x = 2.0_f32;
        assert_eq!(x.abs(), 2.0);
        assert!((x.sqrt() - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert_eq!(f32::precision(), Precision::F32);
    }

    #[test]
    fn test_scalar_f64() {
        assert_eq!(f64::zero(), 0.0_f64);
        assert_eq!(f64::one(), 1.0_f64);
        assert!(f64::infinity().is_infinite());
        assert!(f64::nan().is_nan());

        let x = 2.0_f64;
        assert_eq!(x.abs(), 2.0);
        assert!((x.sqrt() - std::f64::consts::SQRT_2).abs() < 1e-10);
        assert_eq!(f64::precision(), Precision::F64);
    }

    #[test]
    fn test_scalar_conversions() {
        let x_f64 = std::f64::consts::PI;
        let x_f32 = f32::from_f64(x_f64);
        let back_to_f64 = x_f32.to_f64();

        assert!((x_f32 - std::f32::consts::PI).abs() < 1e-5);
        assert!((back_to_f64 - x_f64).abs() < 1e-5);
    }
}
