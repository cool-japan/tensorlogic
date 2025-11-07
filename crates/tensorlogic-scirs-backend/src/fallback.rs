//! Fallback mechanisms for numerical stability.
//!
//! This module provides utilities to handle NaN, Inf, and other numerical
//! issues gracefully during tensor operations.

use crate::error::{NumericalErrorKind, TlBackendError, TlBackendResult};
use scirs2_core::ndarray::ArrayD;

/// Configuration for fallback behavior
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Replace NaN with this value
    pub nan_replacement: f64,
    /// Replace positive infinity with this value
    pub pos_inf_replacement: f64,
    /// Replace negative infinity with this value
    pub neg_inf_replacement: f64,
    /// Whether to fail on NaN (if false, replace)
    pub fail_on_nan: bool,
    /// Whether to fail on Inf (if false, replace)
    pub fail_on_inf: bool,
    /// Minimum value (clamp below this)
    pub min_value: Option<f64>,
    /// Maximum value (clamp above this)
    pub max_value: Option<f64>,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            nan_replacement: 0.0,
            pos_inf_replacement: 1e10,
            neg_inf_replacement: -1e10,
            fail_on_nan: false,
            fail_on_inf: false,
            min_value: None,
            max_value: None,
        }
    }
}

impl FallbackConfig {
    /// Create a strict config that fails on any numerical issue
    pub fn strict() -> Self {
        Self {
            fail_on_nan: true,
            fail_on_inf: true,
            ..Default::default()
        }
    }

    /// Create a permissive config that replaces all invalid values
    pub fn permissive() -> Self {
        Self {
            fail_on_nan: false,
            fail_on_inf: false,
            ..Default::default()
        }
    }

    /// Set NaN replacement value
    pub fn with_nan_replacement(mut self, value: f64) -> Self {
        self.nan_replacement = value;
        self
    }

    /// Set infinity replacement values
    pub fn with_inf_replacement(mut self, pos: f64, neg: f64) -> Self {
        self.pos_inf_replacement = pos;
        self.neg_inf_replacement = neg;
        self
    }

    /// Set value clamping range
    pub fn with_clamp(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }
}

/// Check and potentially fix numerical issues in a tensor
pub fn sanitize_tensor(
    tensor: &ArrayD<f64>,
    config: &FallbackConfig,
    location: &str,
) -> TlBackendResult<ArrayD<f64>> {
    let mut result = tensor.clone();

    // Check for NaN and Inf
    for value in result.iter_mut() {
        if value.is_nan() {
            if config.fail_on_nan {
                return Err(TlBackendError::numerical(NumericalErrorKind::NaN, location));
            }
            *value = config.nan_replacement;
        } else if value.is_infinite() {
            if config.fail_on_inf {
                return Err(TlBackendError::numerical(
                    NumericalErrorKind::Infinity,
                    location,
                ));
            }
            *value = if *value > 0.0 {
                config.pos_inf_replacement
            } else {
                config.neg_inf_replacement
            };
        }

        // Apply clamping if configured
        if let Some(min) = config.min_value {
            if *value < min {
                *value = min;
            }
        }
        if let Some(max) = config.max_value {
            if *value > max {
                *value = max;
            }
        }
    }

    Ok(result)
}

/// Check if a tensor contains any NaN values
pub fn contains_nan(tensor: &ArrayD<f64>) -> bool {
    tensor.iter().any(|v| v.is_nan())
}

/// Check if a tensor contains any infinite values
pub fn contains_inf(tensor: &ArrayD<f64>) -> bool {
    tensor.iter().any(|v| v.is_infinite())
}

/// Check if a tensor is numerically valid (no NaN or Inf)
pub fn is_valid(tensor: &ArrayD<f64>) -> bool {
    !contains_nan(tensor) && !contains_inf(tensor)
}

/// Replace NaN values with a specific value
pub fn replace_nan(tensor: &ArrayD<f64>, replacement: f64) -> ArrayD<f64> {
    tensor.mapv(|v| if v.is_nan() { replacement } else { v })
}

/// Replace infinite values with finite values
pub fn replace_inf(
    tensor: &ArrayD<f64>,
    pos_replacement: f64,
    neg_replacement: f64,
) -> ArrayD<f64> {
    tensor.mapv(|v| {
        if v.is_infinite() {
            if v > 0.0 {
                pos_replacement
            } else {
                neg_replacement
            }
        } else {
            v
        }
    })
}

/// Clamp tensor values to a range
pub fn clamp(tensor: &ArrayD<f64>, min: f64, max: f64) -> ArrayD<f64> {
    tensor.mapv(|v| v.max(min).min(max))
}

/// Safe division that avoids division by zero
pub fn safe_div(a: &ArrayD<f64>, b: &ArrayD<f64>, epsilon: f64) -> ArrayD<f64> {
    let b_safe = b.mapv(|v| {
        if v.abs() < epsilon {
            epsilon * v.signum()
        } else {
            v
        }
    });
    a / &b_safe
}

/// Safe logarithm that avoids log(0)
pub fn safe_log(tensor: &ArrayD<f64>, epsilon: f64) -> ArrayD<f64> {
    tensor.mapv(|v| (v.max(epsilon)).ln())
}

/// Safe square root that avoids sqrt(negative)
pub fn safe_sqrt(tensor: &ArrayD<f64>) -> ArrayD<f64> {
    tensor.mapv(|v| v.max(0.0).sqrt())
}

/// Detect and report numerical issues in a tensor
pub fn detect_issues(tensor: &ArrayD<f64>) -> Vec<NumericalIssue> {
    let mut issues = Vec::new();

    let nan_count = tensor.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        issues.push(NumericalIssue {
            kind: NumericalErrorKind::NaN,
            count: nan_count,
            percentage: (nan_count as f64 / tensor.len() as f64) * 100.0,
        });
    }

    let inf_count = tensor.iter().filter(|v| v.is_infinite()).count();
    if inf_count > 0 {
        issues.push(NumericalIssue {
            kind: NumericalErrorKind::Infinity,
            count: inf_count,
            percentage: (inf_count as f64 / tensor.len() as f64) * 100.0,
        });
    }

    // Check for potential overflow (very large values)
    let large_count = tensor
        .iter()
        .filter(|v| v.abs() > 1e100 && v.is_finite())
        .count();
    if large_count > 0 {
        issues.push(NumericalIssue {
            kind: NumericalErrorKind::Overflow,
            count: large_count,
            percentage: (large_count as f64 / tensor.len() as f64) * 100.0,
        });
    }

    // Check for potential underflow (very small values)
    let small_count = tensor
        .iter()
        .filter(|v| v.abs() < 1e-100 && **v != 0.0)
        .count();
    if small_count > 0 {
        issues.push(NumericalIssue {
            kind: NumericalErrorKind::Underflow,
            count: small_count,
            percentage: (small_count as f64 / tensor.len() as f64) * 100.0,
        });
    }

    issues
}

/// Description of a numerical issue found in a tensor
#[derive(Debug, Clone)]
pub struct NumericalIssue {
    /// Type of issue
    pub kind: NumericalErrorKind,
    /// Number of affected values
    pub count: usize,
    /// Percentage of tensor affected
    pub percentage: f64,
}

impl std::fmt::Display for NumericalIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}: {} values ({:.2}%)",
            self.kind, self.count, self.percentage
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_contains_nan() {
        let valid = array![1.0, 2.0, 3.0].into_dyn();
        assert!(!contains_nan(&valid));

        let invalid = array![1.0, f64::NAN, 3.0].into_dyn();
        assert!(contains_nan(&invalid));
    }

    #[test]
    fn test_contains_inf() {
        let valid = array![1.0, 2.0, 3.0].into_dyn();
        assert!(!contains_inf(&valid));

        let invalid = array![1.0, f64::INFINITY, 3.0].into_dyn();
        assert!(contains_inf(&invalid));
    }

    #[test]
    fn test_is_valid() {
        let valid = array![1.0, 2.0, 3.0].into_dyn();
        assert!(is_valid(&valid));

        let nan_tensor = array![1.0, f64::NAN, 3.0].into_dyn();
        assert!(!is_valid(&nan_tensor));

        let inf_tensor = array![1.0, f64::INFINITY, 3.0].into_dyn();
        assert!(!is_valid(&inf_tensor));
    }

    #[test]
    fn test_replace_nan() {
        let tensor = array![1.0, f64::NAN, 3.0, f64::NAN].into_dyn();
        let result = replace_nan(&tensor, 0.0);

        assert_eq!(result[[0]], 1.0);
        assert_eq!(result[[1]], 0.0);
        assert_eq!(result[[2]], 3.0);
        assert_eq!(result[[3]], 0.0);
    }

    #[test]
    fn test_replace_inf() {
        let tensor = array![1.0, f64::INFINITY, -3.0, f64::NEG_INFINITY].into_dyn();
        let result = replace_inf(&tensor, 100.0, -100.0);

        assert_eq!(result[[0]], 1.0);
        assert_eq!(result[[1]], 100.0);
        assert_eq!(result[[2]], -3.0);
        assert_eq!(result[[3]], -100.0);
    }

    #[test]
    fn test_clamp() {
        let tensor = array![-5.0, 0.0, 5.0, 10.0].into_dyn();
        let result = clamp(&tensor, -2.0, 7.0);

        assert_eq!(result[[0]], -2.0);
        assert_eq!(result[[1]], 0.0);
        assert_eq!(result[[2]], 5.0);
        assert_eq!(result[[3]], 7.0);
    }

    #[test]
    fn test_sanitize_tensor_permissive() {
        let tensor = array![1.0, f64::NAN, f64::INFINITY, -3.0].into_dyn();
        let config = FallbackConfig::permissive();
        let result = sanitize_tensor(&tensor, &config, "test").unwrap();

        assert_eq!(result[[0]], 1.0);
        assert_eq!(result[[1]], 0.0); // NaN replaced with 0.0
        assert_eq!(result[[2]], 1e10); // Inf replaced
        assert_eq!(result[[3]], -3.0);
    }

    #[test]
    fn test_sanitize_tensor_strict() {
        let tensor = array![1.0, f64::NAN, 3.0].into_dyn();
        let config = FallbackConfig::strict();
        let result = sanitize_tensor(&tensor, &config, "test");

        assert!(result.is_err());
    }

    #[test]
    fn test_safe_div() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let b = array![2.0, 0.0, 4.0].into_dyn();
        let result = safe_div(&a, &b, 1e-10);

        assert_eq!(result[[0]], 0.5);
        assert!(result[[1]].is_finite()); // Should not be Inf
        assert_eq!(result[[2]], 0.75);
    }

    #[test]
    fn test_safe_log() {
        let tensor = array![1.0, 0.0, 10.0].into_dyn();
        let result = safe_log(&tensor, 1e-10);

        assert_eq!(result[[0]], 0.0);
        assert!(result[[1]].is_finite()); // Should not be -Inf
        assert!((result[[2]] - 10.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_safe_sqrt() {
        let tensor = array![4.0, -1.0, 9.0].into_dyn();
        let result = safe_sqrt(&tensor);

        assert_eq!(result[[0]], 2.0);
        assert_eq!(result[[1]], 0.0); // Negative treated as 0
        assert_eq!(result[[2]], 3.0);
    }

    #[test]
    fn test_detect_issues() {
        let tensor = array![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0, f64::NAN, 7.0, 8.0].into_dyn();

        let issues = detect_issues(&tensor);

        assert!(issues
            .iter()
            .any(|i| matches!(i.kind, NumericalErrorKind::NaN)));
        assert!(issues
            .iter()
            .any(|i| matches!(i.kind, NumericalErrorKind::Infinity)));

        let nan_issue = issues
            .iter()
            .find(|i| matches!(i.kind, NumericalErrorKind::NaN))
            .unwrap();
        assert_eq!(nan_issue.count, 2);
    }

    #[test]
    fn test_fallback_config_builder() {
        let config = FallbackConfig::default()
            .with_nan_replacement(1.0)
            .with_inf_replacement(1e5, -1e5)
            .with_clamp(-100.0, 100.0);

        assert_eq!(config.nan_replacement, 1.0);
        assert_eq!(config.pos_inf_replacement, 1e5);
        assert_eq!(config.neg_inf_replacement, -1e5);
        assert_eq!(config.min_value, Some(-100.0));
        assert_eq!(config.max_value, Some(100.0));
    }
}
