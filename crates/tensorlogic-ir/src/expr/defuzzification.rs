//! Defuzzification methods for converting fuzzy sets to crisp values.
//!
//! This module implements various defuzzification strategies used to convert
//! fuzzy membership functions into concrete numerical values, essential for:
//! - Fuzzy control systems
//! - Decision-making under uncertainty
//! - Fuzzy inference system outputs
//!
//! # Defuzzification Methods
//!
//! - **Centroid (COA)**: Center of Area/Gravity - most common method
//! - **Bisector**: Vertical line dividing the area into two equal parts
//! - **Mean of Maximum (MOM)**: Average of maximum membership values
//! - **Smallest of Maximum (SOM)**: Leftmost point of maximum membership
//! - **Largest of Maximum (LOM)**: Rightmost point of maximum membership
//! - **Weighted Average**: For discrete/singleton fuzzy sets

use std::collections::HashMap;

/// Defuzzification method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefuzzificationMethod {
    /// Center of Area (Centroid) - most widely used
    Centroid,
    /// Bisector of Area
    Bisector,
    /// Mean of Maximum
    MeanOfMaximum,
    /// Smallest (leftmost) of Maximum
    SmallestOfMaximum,
    /// Largest (rightmost) of Maximum
    LargestOfMaximum,
    /// Weighted Average (for singleton fuzzy sets)
    WeightedAverage,
}

/// Represents a fuzzy membership function over a continuous domain.
///
/// Discretized as samples over [min, max] range.
#[derive(Debug, Clone)]
pub struct FuzzySet {
    /// Domain minimum value
    pub min: f64,
    /// Domain maximum value
    pub max: f64,
    /// Membership values at sample points (uniformly spaced)
    pub memberships: Vec<f64>,
}

impl FuzzySet {
    /// Create a new fuzzy set with given range and sample size.
    pub fn new(min: f64, max: f64, samples: usize) -> Self {
        Self {
            min,
            max,
            memberships: vec![0.0; samples],
        }
    }

    /// Create a fuzzy set from explicit membership values.
    pub fn from_memberships(min: f64, max: f64, memberships: Vec<f64>) -> Self {
        Self {
            min,
            max,
            memberships,
        }
    }

    /// Get the domain value at a given sample index.
    fn value_at_index(&self, index: usize) -> f64 {
        let range = self.max - self.min;
        let step = range / (self.memberships.len() - 1).max(1) as f64;
        self.min + index as f64 * step
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.memberships.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.memberships.is_empty()
    }

    /// Find maximum membership value.
    fn max_membership(&self) -> f64 {
        self.memberships.iter().fold(0.0f64, |a, &b| a.max(b))
    }

    /// Find indices of maximum membership.
    fn max_membership_indices(&self) -> Vec<usize> {
        let max_val = self.max_membership();
        if max_val == 0.0 {
            return vec![];
        }

        self.memberships
            .iter()
            .enumerate()
            .filter(|(_, &val)| (val - max_val).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute area under the membership function (using trapezoidal rule).
    fn area(&self) -> f64 {
        if self.memberships.is_empty() {
            return 0.0;
        }

        let step = (self.max - self.min) / (self.memberships.len() - 1).max(1) as f64;
        let mut area = 0.0;

        for i in 0..self.memberships.len() - 1 {
            // Trapezoidal rule: (h/2) * (f(x_i) + f(x_{i+1}))
            area += step * (self.memberships[i] + self.memberships[i + 1]) / 2.0;
        }

        area
    }

    /// Compute centroid of the fuzzy set.
    fn centroid_numerator(&self) -> f64 {
        if self.memberships.is_empty() {
            return 0.0;
        }

        let step = (self.max - self.min) / (self.memberships.len() - 1).max(1) as f64;
        let mut numerator = 0.0;

        for i in 0..self.memberships.len() - 1 {
            let x1 = self.value_at_index(i);
            let x2 = self.value_at_index(i + 1);
            let y1 = self.memberships[i];
            let y2 = self.memberships[i + 1];

            // Centroid of trapezoid: x_c = (x1 + x2)/2
            let x_mid = (x1 + x2) / 2.0;
            let trap_area = step * (y1 + y2) / 2.0;

            numerator += x_mid * trap_area;
        }

        numerator
    }
}

/// Defuzzify a fuzzy set using the specified method.
///
/// Returns the crisp output value, or None if the fuzzy set is empty
/// or the method cannot be applied.
pub fn defuzzify(fuzzy_set: &FuzzySet, method: DefuzzificationMethod) -> Option<f64> {
    match method {
        DefuzzificationMethod::Centroid => centroid(fuzzy_set),
        DefuzzificationMethod::Bisector => bisector(fuzzy_set),
        DefuzzificationMethod::MeanOfMaximum => mean_of_maximum(fuzzy_set),
        DefuzzificationMethod::SmallestOfMaximum => smallest_of_maximum(fuzzy_set),
        DefuzzificationMethod::LargestOfMaximum => largest_of_maximum(fuzzy_set),
        DefuzzificationMethod::WeightedAverage => weighted_average(fuzzy_set),
    }
}

/// Centroid (Center of Area) defuzzification.
///
/// Computes: ∫x·μ(x)dx / ∫μ(x)dx
pub fn centroid(fuzzy_set: &FuzzySet) -> Option<f64> {
    let area = fuzzy_set.area();
    if area == 0.0 {
        return None;
    }

    let numerator = fuzzy_set.centroid_numerator();
    Some(numerator / area)
}

/// Bisector of Area defuzzification.
///
/// Finds the x value that divides the area under the membership
/// function into two equal parts.
pub fn bisector(fuzzy_set: &FuzzySet) -> Option<f64> {
    let total_area = fuzzy_set.area();
    if total_area == 0.0 {
        return None;
    }

    let target_area = total_area / 2.0;
    let step = (fuzzy_set.max - fuzzy_set.min) / (fuzzy_set.memberships.len() - 1).max(1) as f64;

    let mut cumulative_area = 0.0;

    for i in 0..fuzzy_set.memberships.len() - 1 {
        let trap_area = step * (fuzzy_set.memberships[i] + fuzzy_set.memberships[i + 1]) / 2.0;
        if cumulative_area + trap_area >= target_area {
            // Linear interpolation within this segment
            let remaining = target_area - cumulative_area;
            let fraction = remaining / trap_area;
            let x = fuzzy_set.value_at_index(i)
                + fraction * (fuzzy_set.value_at_index(i + 1) - fuzzy_set.value_at_index(i));
            return Some(x);
        }
        cumulative_area += trap_area;
    }

    // Shouldn't reach here, but return midpoint as fallback
    Some((fuzzy_set.min + fuzzy_set.max) / 2.0)
}

/// Mean of Maximum defuzzification.
///
/// Returns the average of all x values where membership is maximum.
pub fn mean_of_maximum(fuzzy_set: &FuzzySet) -> Option<f64> {
    let max_indices = fuzzy_set.max_membership_indices();
    if max_indices.is_empty() {
        return None;
    }

    let sum: f64 = max_indices
        .iter()
        .map(|&i| fuzzy_set.value_at_index(i))
        .sum();

    Some(sum / max_indices.len() as f64)
}

/// Smallest of Maximum defuzzification.
///
/// Returns the leftmost x value where membership is maximum.
pub fn smallest_of_maximum(fuzzy_set: &FuzzySet) -> Option<f64> {
    let max_indices = fuzzy_set.max_membership_indices();
    max_indices.first().map(|&i| fuzzy_set.value_at_index(i))
}

/// Largest of Maximum defuzzification.
///
/// Returns the rightmost x value where membership is maximum.
pub fn largest_of_maximum(fuzzy_set: &FuzzySet) -> Option<f64> {
    let max_indices = fuzzy_set.max_membership_indices();
    max_indices.last().map(|&i| fuzzy_set.value_at_index(i))
}

/// Weighted Average defuzzification for singleton fuzzy sets.
///
/// Computes: Σ(x_i * μ(x_i)) / Σμ(x_i)
pub fn weighted_average(fuzzy_set: &FuzzySet) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &membership) in fuzzy_set.memberships.iter().enumerate() {
        let x = fuzzy_set.value_at_index(i);
        numerator += x * membership;
        denominator += membership;
    }

    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

/// Singleton fuzzy set for discrete inputs (common in fuzzy control).
///
/// Represents a collection of singleton (crisp input, fuzzy membership) pairs.
#[derive(Debug, Clone)]
pub struct SingletonFuzzySet {
    /// Map from crisp value to membership degree
    pub singletons: HashMap<String, f64>,
}

impl SingletonFuzzySet {
    /// Create a new empty singleton fuzzy set.
    pub fn new() -> Self {
        Self {
            singletons: HashMap::new(),
        }
    }

    /// Add a singleton (value, membership) pair.
    pub fn add(&mut self, value: String, membership: f64) {
        self.singletons.insert(value, membership.clamp(0.0, 1.0));
    }

    /// Defuzzify using weighted average of singletons.
    ///
    /// Assumes values can be parsed as f64.
    pub fn defuzzify(&self) -> Option<f64> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (value_str, &membership) in &self.singletons {
            if let Ok(value) = value_str.parse::<f64>() {
                numerator += value * membership;
                denominator += membership;
            }
        }

        if denominator == 0.0 {
            None
        } else {
            Some(numerator / denominator)
        }
    }

    /// Get the singleton with maximum membership (winner-takes-all).
    pub fn winner_takes_all(&self) -> Option<(String, f64)> {
        self.singletons
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, &v)| (k.clone(), v))
    }
}

impl Default for SingletonFuzzySet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fuzzy_set() -> FuzzySet {
        // Triangular membership function peaked at 0.5
        FuzzySet::from_memberships(
            0.0,
            1.0,
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0],
        )
    }

    #[test]
    fn test_fuzzy_set_creation() {
        let fs = FuzzySet::new(0.0, 10.0, 11);
        assert_eq!(fs.len(), 11);
        assert!((fs.min - 0.0).abs() < 1e-10);
        assert!((fs.max - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_value_at_index() {
        let fs = FuzzySet::new(0.0, 10.0, 11);
        assert!((fs.value_at_index(0) - 0.0).abs() < 1e-10);
        assert!((fs.value_at_index(5) - 5.0).abs() < 1e-10);
        assert!((fs.value_at_index(10) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_membership() {
        let fs = create_test_fuzzy_set();
        assert!((fs.max_membership() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_membership_indices() {
        let fs = create_test_fuzzy_set();
        let indices = fs.max_membership_indices();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 4);
    }

    #[test]
    fn test_centroid() {
        let fs = create_test_fuzzy_set();
        let result = centroid(&fs).unwrap();
        // For symmetric triangular, centroid should be near 0.5
        assert!((result - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_bisector() {
        let fs = create_test_fuzzy_set();
        let result = bisector(&fs).unwrap();
        // Bisector for symmetric shape should be near center
        assert!((result - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_mean_of_maximum() {
        let fs = create_test_fuzzy_set();
        let result = mean_of_maximum(&fs).unwrap();
        // MOM for single peak at 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smallest_of_maximum() {
        let fs = FuzzySet::from_memberships(0.0, 1.0, vec![0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]);
        let result = smallest_of_maximum(&fs).unwrap();
        // Leftmost max at index 2 → 2/6 ≈ 0.33
        assert!((result - 0.333).abs() < 0.05);
    }

    #[test]
    fn test_largest_of_maximum() {
        let fs = FuzzySet::from_memberships(0.0, 1.0, vec![0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]);
        let result = largest_of_maximum(&fs).unwrap();
        // Rightmost max at index 4 → 4/6 ≈ 0.67
        assert!((result - 0.667).abs() < 0.05);
    }

    #[test]
    fn test_weighted_average() {
        let fs = FuzzySet::from_memberships(0.0, 10.0, vec![0.2, 0.5, 0.8, 0.5, 0.2]);
        let result = weighted_average(&fs).unwrap();
        // Should be weighted toward middle (index 2)
        assert!(result > 4.0 && result < 6.0);
    }

    #[test]
    fn test_empty_fuzzy_set() {
        let fs = FuzzySet::from_memberships(0.0, 1.0, vec![0.0, 0.0, 0.0]);
        assert!(centroid(&fs).is_none());
        assert!(bisector(&fs).is_none());
        assert!(mean_of_maximum(&fs).is_none());
    }

    #[test]
    fn test_singleton_fuzzy_set() {
        let mut sfs = SingletonFuzzySet::new();
        sfs.add("0.0".to_string(), 0.2);
        sfs.add("5.0".to_string(), 0.8);
        sfs.add("10.0".to_string(), 0.3);

        let result = sfs.defuzzify().unwrap();
        // Weighted average: (0*0.2 + 5*0.8 + 10*0.3) / (0.2 + 0.8 + 0.3)
        // = (0 + 4 + 3) / 1.3 ≈ 5.38
        assert!((result - 5.38).abs() < 0.1);
    }

    #[test]
    fn test_singleton_winner_takes_all() {
        let mut sfs = SingletonFuzzySet::new();
        sfs.add("low".to_string(), 0.3);
        sfs.add("medium".to_string(), 0.8);
        sfs.add("high".to_string(), 0.5);

        let (winner, membership) = sfs.winner_takes_all().unwrap();
        assert_eq!(winner, "medium");
        assert!((membership - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_defuzzify_dispatch() {
        let fs = create_test_fuzzy_set();

        assert!(defuzzify(&fs, DefuzzificationMethod::Centroid).is_some());
        assert!(defuzzify(&fs, DefuzzificationMethod::Bisector).is_some());
        assert!(defuzzify(&fs, DefuzzificationMethod::MeanOfMaximum).is_some());
        assert!(defuzzify(&fs, DefuzzificationMethod::SmallestOfMaximum).is_some());
        assert!(defuzzify(&fs, DefuzzificationMethod::LargestOfMaximum).is_some());
        assert!(defuzzify(&fs, DefuzzificationMethod::WeightedAverage).is_some());
    }
}
