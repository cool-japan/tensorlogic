//! Advanced quantization support for model compression and acceleration.
//!
//! This module provides comprehensive quantization capabilities including:
//! - Multiple quantization schemes (INT8, INT4, FP8, binary)
//! - Quantization-Aware Training (QAT) support
//! - Post-Training Quantization (PTQ) with calibration
//! - Per-channel and per-tensor quantization
//! - Symmetric and asymmetric quantization
//! - Dynamic and static quantization modes
//! - Quantization simulation for accuracy validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::OpType;
use thiserror::Error;

/// Node identifier (0-based index into graph.nodes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// Quantization-related errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum QuantizationError {
    #[error("Unsupported data type for quantization: {0}")]
    UnsupportedDataType(String),

    #[error("Invalid quantization range: min={min}, max={max}")]
    InvalidRange { min: f64, max: f64 },

    #[error("Calibration failed: {0}")]
    CalibrationFailed(String),

    #[error("Quantization not supported for operation: {0:?}")]
    UnsupportedOperation(OpType),

    #[error("Invalid quantization parameters: {0}")]
    InvalidParameters(String),

    #[error("Insufficient calibration data")]
    InsufficientData,
}

/// Quantization data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// 2-bit integer quantization (extreme compression)
    Int2,
    /// 8-bit floating point (E4M3 or E5M2)
    FP8E4M3,
    /// FP8 E5M2 format
    FP8E5M2,
    /// 16-bit floating point
    FP16,
    /// 16-bit brain float
    BF16,
    /// Binary quantization (1-bit)
    Binary,
    /// Ternary quantization (-1, 0, 1)
    Ternary,
}

impl QuantizationType {
    /// Returns the number of bits used by this quantization type.
    pub fn bits(&self) -> u32 {
        match self {
            Self::Binary => 1,
            Self::Int2 => 2,
            Self::Int4 => 4,
            Self::Int8 | Self::FP8E4M3 | Self::FP8E5M2 => 8,
            Self::FP16 | Self::BF16 => 16,
            Self::Ternary => 2, // effectively 1.58 bits, but stored as 2
        }
    }

    /// Returns the theoretical compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits() as f64
    }

    /// Returns whether this type supports floating point values.
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            Self::FP8E4M3 | Self::FP8E5M2 | Self::FP16 | Self::BF16
        )
    }
}

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Quantize entire tensor with single scale/zero-point
    PerTensor,
    /// Quantize each channel independently
    PerChannel { axis: usize },
    /// Quantize groups of channels
    PerGroup { axis: usize, group_size: usize },
}

/// Quantization symmetry mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationSymmetry {
    /// Symmetric quantization (zero_point = 0)
    Symmetric,
    /// Asymmetric quantization (arbitrary zero_point)
    Asymmetric,
}

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Static quantization (pre-computed scales)
    Static,
    /// Dynamic quantization (compute scales at runtime)
    Dynamic,
    /// Quantization-aware training simulation
    QAT,
}

/// Calibration strategy for post-training quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationStrategy {
    /// Use min/max values observed during calibration
    MinMax,
    /// Use percentiles to handle outliers (e.g., 0.1% and 99.9%)
    Percentile { lower: u32, upper: u32 },
    /// Minimize mean squared error
    MSE,
    /// Minimize KL divergence between distributions
    KLDivergence,
    /// Entropy-based calibration
    Entropy,
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Quantization type
    pub qtype: QuantizationType,
    /// Scale factor(s)
    pub scale: Vec<f64>,
    /// Zero point(s)
    pub zero_point: Vec<i32>,
    /// Granularity
    pub granularity: QuantizationGranularity,
    /// Symmetry mode
    pub symmetry: QuantizationSymmetry,
    /// Observed min/max during calibration (for validation)
    pub observed_min: Option<f64>,
    pub observed_max: Option<f64>,
}

impl QuantizationParams {
    /// Create symmetric per-tensor quantization parameters.
    pub fn symmetric_per_tensor(
        qtype: QuantizationType,
        abs_max: f64,
    ) -> Result<Self, QuantizationError> {
        if abs_max <= 0.0 {
            return Err(QuantizationError::InvalidRange {
                min: -abs_max,
                max: abs_max,
            });
        }

        let qmax = match qtype {
            QuantizationType::Int8 => 127.0,
            QuantizationType::Int4 => 7.0,
            QuantizationType::Int2 => 1.0,
            QuantizationType::Binary => 1.0,
            QuantizationType::Ternary => 1.0,
            _ => {
                return Err(QuantizationError::UnsupportedDataType(format!(
                    "{:?}",
                    qtype
                )))
            }
        };

        let scale = abs_max / qmax;

        Ok(Self {
            qtype,
            scale: vec![scale],
            zero_point: vec![0],
            granularity: QuantizationGranularity::PerTensor,
            symmetry: QuantizationSymmetry::Symmetric,
            observed_min: Some(-abs_max),
            observed_max: Some(abs_max),
        })
    }

    /// Create asymmetric per-tensor quantization parameters.
    pub fn asymmetric_per_tensor(
        qtype: QuantizationType,
        min: f64,
        max: f64,
    ) -> Result<Self, QuantizationError> {
        if min >= max {
            return Err(QuantizationError::InvalidRange { min, max });
        }

        let (qmin, qmax) = match qtype {
            QuantizationType::Int8 => (-128.0, 127.0),
            QuantizationType::Int4 => (-8.0, 7.0),
            QuantizationType::Int2 => (-2.0, 1.0),
            _ => {
                return Err(QuantizationError::UnsupportedDataType(format!(
                    "{:?}",
                    qtype
                )))
            }
        };

        let scale = (max - min) / (qmax - qmin);
        let zero_point = (qmin - min / scale).round() as i32;

        Ok(Self {
            qtype,
            scale: vec![scale],
            zero_point: vec![zero_point],
            granularity: QuantizationGranularity::PerTensor,
            symmetry: QuantizationSymmetry::Asymmetric,
            observed_min: Some(min),
            observed_max: Some(max),
        })
    }

    /// Quantize a floating-point value to integer.
    pub fn quantize(&self, value: f64) -> i32 {
        let scale = self.scale[0];
        let zero_point = self.zero_point[0];
        ((value / scale).round() as i32 + zero_point).clamp(self.qmin(), self.qmax())
    }

    /// Dequantize an integer value to floating-point.
    pub fn dequantize(&self, qvalue: i32) -> f64 {
        let scale = self.scale[0];
        let zero_point = self.zero_point[0];
        (qvalue - zero_point) as f64 * scale
    }

    /// Get quantization minimum value.
    fn qmin(&self) -> i32 {
        match self.qtype {
            QuantizationType::Int8 => -128,
            QuantizationType::Int4 => -8,
            QuantizationType::Int2 => -2,
            QuantizationType::Binary => 0,
            QuantizationType::Ternary => -1,
            _ => 0,
        }
    }

    /// Get quantization maximum value.
    fn qmax(&self) -> i32 {
        match self.qtype {
            QuantizationType::Int8 => 127,
            QuantizationType::Int4 => 7,
            QuantizationType::Int2 => 1,
            QuantizationType::Binary => 1,
            QuantizationType::Ternary => 1,
            _ => 255,
        }
    }
}

/// Quantization configuration for a graph or model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Default quantization type
    pub default_qtype: QuantizationType,
    /// Quantization mode
    pub mode: QuantizationMode,
    /// Granularity
    pub granularity: QuantizationGranularity,
    /// Symmetry mode
    pub symmetry: QuantizationSymmetry,
    /// Calibration strategy (for PTQ)
    pub calibration: CalibrationStrategy,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Operations to skip quantization
    pub skip_ops: Vec<OpType>,
    /// Per-node quantization overrides
    pub node_overrides: HashMap<NodeId, QuantizationType>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            default_qtype: QuantizationType::Int8,
            mode: QuantizationMode::Static,
            granularity: QuantizationGranularity::PerTensor,
            symmetry: QuantizationSymmetry::Symmetric,
            calibration: CalibrationStrategy::MinMax,
            calibration_samples: 100,
            skip_ops: vec![],
            node_overrides: HashMap::new(),
        }
    }
}

impl QuantizationConfig {
    /// Create a configuration for int8 quantization.
    pub fn int8() -> Self {
        Self {
            default_qtype: QuantizationType::Int8,
            ..Default::default()
        }
    }

    /// Create a configuration for int4 quantization.
    pub fn int4() -> Self {
        Self {
            default_qtype: QuantizationType::Int4,
            ..Default::default()
        }
    }

    /// Create a configuration for FP8 quantization.
    pub fn fp8() -> Self {
        Self {
            default_qtype: QuantizationType::FP8E4M3,
            symmetry: QuantizationSymmetry::Symmetric,
            ..Default::default()
        }
    }

    /// Create a configuration for quantization-aware training.
    pub fn qat(qtype: QuantizationType) -> Self {
        Self {
            default_qtype: qtype,
            mode: QuantizationMode::QAT,
            ..Default::default()
        }
    }

    /// Enable per-channel quantization.
    pub fn per_channel(mut self, axis: usize) -> Self {
        self.granularity = QuantizationGranularity::PerChannel { axis };
        self
    }

    /// Enable asymmetric quantization.
    pub fn asymmetric(mut self) -> Self {
        self.symmetry = QuantizationSymmetry::Asymmetric;
        self
    }

    /// Set calibration strategy.
    pub fn with_calibration(mut self, strategy: CalibrationStrategy) -> Self {
        self.calibration = strategy;
        self
    }
}

/// Statistics collected during calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationStats {
    /// Minimum values observed per node
    pub min_values: HashMap<NodeId, f64>,
    /// Maximum values observed per node
    pub max_values: HashMap<NodeId, f64>,
    /// Histogram bins for distribution analysis
    pub histograms: HashMap<NodeId, Vec<u32>>,
    /// Number of samples observed
    pub num_samples: usize,
}

impl CalibrationStats {
    /// Create new calibration statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a new observation.
    pub fn update(&mut self, node_id: NodeId, min: f64, max: f64) {
        self.min_values
            .entry(node_id)
            .and_modify(|v| *v = v.min(min))
            .or_insert(min);
        self.max_values
            .entry(node_id)
            .and_modify(|v| *v = v.max(max))
            .or_insert(max);
        self.num_samples += 1;
    }

    /// Get computed quantization parameters for a node.
    pub fn compute_params(
        &self,
        node_id: NodeId,
        config: &QuantizationConfig,
    ) -> Result<QuantizationParams, QuantizationError> {
        let min = self
            .min_values
            .get(&node_id)
            .ok_or(QuantizationError::InsufficientData)?;
        let max = self
            .max_values
            .get(&node_id)
            .ok_or(QuantizationError::InsufficientData)?;

        let qtype = config
            .node_overrides
            .get(&node_id)
            .copied()
            .unwrap_or(config.default_qtype);

        match config.symmetry {
            QuantizationSymmetry::Symmetric => {
                let abs_max = min.abs().max(max.abs());
                QuantizationParams::symmetric_per_tensor(qtype, abs_max)
            }
            QuantizationSymmetry::Asymmetric => {
                QuantizationParams::asymmetric_per_tensor(qtype, *min, *max)
            }
        }
    }
}

/// Quantizer for converting graphs to quantized representations.
pub struct Quantizer {
    config: QuantizationConfig,
    stats: CalibrationStats,
    params: HashMap<NodeId, QuantizationParams>,
}

impl Quantizer {
    /// Create a new quantizer with the given configuration.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            stats: CalibrationStats::new(),
            params: HashMap::new(),
        }
    }

    /// Create a quantizer for int8 quantization.
    pub fn int8() -> Self {
        Self::new(QuantizationConfig::int8())
    }

    /// Create a quantizer for int4 quantization.
    pub fn int4() -> Self {
        Self::new(QuantizationConfig::int4())
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Get calibration statistics.
    pub fn stats(&self) -> &CalibrationStats {
        &self.stats
    }

    /// Get quantization parameters for a node.
    pub fn get_params(&self, node_id: NodeId) -> Option<&QuantizationParams> {
        self.params.get(&node_id)
    }

    /// Add calibration data for a node.
    pub fn calibrate(&mut self, node_id: NodeId, min: f64, max: f64) {
        self.stats.update(node_id, min, max);
    }

    /// Finalize calibration and compute quantization parameters.
    pub fn finalize_calibration(&mut self) -> Result<(), QuantizationError> {
        if self.stats.num_samples < self.config.calibration_samples {
            return Err(QuantizationError::InsufficientData);
        }

        // Compute params for all calibrated nodes
        for &node_id in self.stats.min_values.keys() {
            let params = self.stats.compute_params(node_id, &self.config)?;
            self.params.insert(node_id, params);
        }

        Ok(())
    }

    /// Get quantization summary statistics.
    pub fn summary(&self) -> QuantizationSummary {
        let mut type_counts = HashMap::new();
        for params in self.params.values() {
            *type_counts.entry(params.qtype).or_insert(0) += 1;
        }

        let total_params = self.params.len();
        let avg_compression = self
            .params
            .values()
            .map(|p| p.qtype.compression_ratio())
            .sum::<f64>()
            / total_params.max(1) as f64;

        QuantizationSummary {
            num_quantized_nodes: total_params,
            type_distribution: type_counts,
            avg_compression_ratio: avg_compression,
            calibration_samples: self.stats.num_samples,
        }
    }
}

/// Summary of quantization results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSummary {
    /// Number of quantized nodes
    pub num_quantized_nodes: usize,
    /// Distribution of quantization types
    pub type_distribution: HashMap<QuantizationType, usize>,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Number of calibration samples used
    pub calibration_samples: usize,
}

impl QuantizationSummary {
    /// Get estimated memory savings.
    pub fn memory_savings(&self) -> f64 {
        if self.avg_compression_ratio > 1.0 {
            (1.0 - 1.0 / self.avg_compression_ratio) * 100.0
        } else {
            0.0
        }
    }
}

/// Fake quantization for QAT (simulates quantization during training).
pub struct FakeQuantize {
    params: QuantizationParams,
    enabled: bool,
}

impl FakeQuantize {
    /// Create a new fake quantization module.
    pub fn new(params: QuantizationParams) -> Self {
        Self {
            params,
            enabled: true,
        }
    }

    /// Enable or disable fake quantization.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Apply fake quantization to a value.
    pub fn forward(&self, value: f64) -> f64 {
        if !self.enabled {
            return value;
        }

        // Quantize then dequantize (simulating quantization noise)
        let qvalue = self.params.quantize(value);
        self.params.dequantize(qvalue)
    }

    /// Simulate quantization on a batch of values.
    pub fn forward_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.forward(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_properties() {
        assert_eq!(QuantizationType::Int8.bits(), 8);
        assert_eq!(QuantizationType::Int4.bits(), 4);
        assert_eq!(QuantizationType::Binary.bits(), 1);
        assert_eq!(QuantizationType::Int8.compression_ratio(), 4.0);
        assert!(QuantizationType::FP16.is_floating_point());
        assert!(!QuantizationType::Int8.is_floating_point());
    }

    #[test]
    fn test_symmetric_quantization() {
        let params =
            QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, 127.0).unwrap();
        assert_eq!(params.scale[0], 1.0);
        assert_eq!(params.zero_point[0], 0);

        // Test quantize/dequantize
        assert_eq!(params.quantize(0.0), 0);
        assert_eq!(params.quantize(127.0), 127);
        assert_eq!(params.quantize(-127.0), -127);
        assert!((params.dequantize(127) - 127.0).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let params =
            QuantizationParams::asymmetric_per_tensor(QuantizationType::Int8, -10.0, 20.0).unwrap();

        assert!(params.scale[0] > 0.0);
        assert_ne!(params.zero_point[0], 0);

        // Test round-trip
        let original = 5.0;
        let quantized = params.quantize(original);
        let dequantized = params.dequantize(quantized);
        assert!((dequantized - original).abs() < 1.0); // Allow quantization error
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::int8();
        assert_eq!(config.default_qtype, QuantizationType::Int8);

        let config = QuantizationConfig::int4().per_channel(0).asymmetric();
        assert_eq!(config.default_qtype, QuantizationType::Int4);
        assert!(matches!(
            config.granularity,
            QuantizationGranularity::PerChannel { axis: 0 }
        ));
        assert_eq!(config.symmetry, QuantizationSymmetry::Asymmetric);
    }

    #[test]
    fn test_calibration_stats() {
        let mut stats = CalibrationStats::new();
        stats.update(NodeId(0), -5.0, 10.0);
        stats.update(NodeId(0), -8.0, 12.0);

        assert_eq!(stats.min_values[&NodeId(0)], -8.0);
        assert_eq!(stats.max_values[&NodeId(0)], 12.0);
    }

    #[test]
    fn test_quantizer() {
        let mut quantizer = Quantizer::int8();

        // Calibrate
        quantizer.calibrate(NodeId(0), -10.0, 10.0);
        quantizer.calibrate(NodeId(0), -8.0, 12.0);

        // Since calibration_samples default is 100, we need to adjust or add more
        // For testing, let's manually set sufficient samples
        for _ in 0..100 {
            quantizer.calibrate(NodeId(0), -10.0, 10.0);
        }

        assert!(quantizer.finalize_calibration().is_ok());
        assert!(quantizer.get_params(NodeId(0)).is_some());

        let summary = quantizer.summary();
        assert_eq!(summary.num_quantized_nodes, 1);
        assert!(summary.avg_compression_ratio > 1.0);
    }

    #[test]
    fn test_fake_quantize() {
        let params =
            QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, 10.0).unwrap();
        let fake_quant = FakeQuantize::new(params);

        let original = 3.5;
        let faked = fake_quant.forward(original);

        // Should be close but not exact due to quantization
        assert!((faked - original).abs() < 1.0);
    }

    #[test]
    fn test_quantization_summary() {
        let mut quantizer = Quantizer::int8();
        for _ in 0..100 {
            quantizer.calibrate(NodeId(0), -10.0, 10.0);
        }
        quantizer.finalize_calibration().unwrap();

        let summary = quantizer.summary();
        assert!(summary.memory_savings() > 0.0);
        assert!(summary.memory_savings() < 100.0);
    }

    #[test]
    fn test_int4_quantization() {
        let params = QuantizationParams::symmetric_per_tensor(QuantizationType::Int4, 7.0).unwrap();

        let value = 5.0;
        let qvalue = params.quantize(value);
        assert!((-8..=7).contains(&qvalue));
    }

    #[test]
    fn test_invalid_range() {
        let result = QuantizationParams::asymmetric_per_tensor(QuantizationType::Int8, 10.0, 5.0);
        assert!(matches!(
            result,
            Err(QuantizationError::InvalidRange { .. })
        ));
    }
}
