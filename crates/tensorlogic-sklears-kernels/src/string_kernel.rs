//! String kernels for text similarity.
//!
//! These kernels measure similarity between text sequences using
//! substring matching, n-grams, and subsequence features.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// N-gram string kernel configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NGramKernelConfig {
    /// N-gram size
    pub n: usize,
    /// Whether to normalize by string length
    pub normalize: bool,
}

impl NGramKernelConfig {
    /// Create configuration with n-gram size
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "n".to_string(),
                value: n.to_string(),
                reason: "n-gram size must be positive".to_string(),
            });
        }

        Ok(Self { n, normalize: true })
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// N-gram string kernel
///
/// Measures similarity by counting common n-grams.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{NGramKernel, NGramKernelConfig};
///
/// let config = NGramKernelConfig::new(2).unwrap(); // bigrams
/// let kernel = NGramKernel::new(config);
///
/// let text1 = "hello world";
/// let text2 = "hello there";
///
/// let sim = kernel.compute_strings(text1, text2).unwrap();
/// println!("Similarity: {}", sim);
/// ```
pub struct NGramKernel {
    config: NGramKernelConfig,
}

impl NGramKernel {
    /// Create a new n-gram kernel
    pub fn new(config: NGramKernelConfig) -> Self {
        Self { config }
    }

    /// Extract n-grams from text
    fn extract_ngrams(&self, text: &str) -> HashMap<String, usize> {
        let mut ngrams = HashMap::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() < self.config.n {
            return ngrams;
        }

        for i in 0..=(chars.len() - self.config.n) {
            let ngram: String = chars[i..i + self.config.n].iter().collect();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }

        ngrams
    }

    /// Compute similarity between two text strings
    pub fn compute_strings(&self, text1: &str, text2: &str) -> Result<f64> {
        let ngrams1 = self.extract_ngrams(text1);
        let ngrams2 = self.extract_ngrams(text2);

        // Compute intersection
        let mut similarity = 0.0;
        for (ngram, count1) in &ngrams1 {
            if let Some(count2) = ngrams2.get(ngram) {
                similarity += (*count1).min(*count2) as f64;
            }
        }

        if self.config.normalize {
            let total1: usize = ngrams1.values().sum();
            let total2: usize = ngrams2.values().sum();
            let normalizer = ((total1 * total2) as f64).sqrt();

            if normalizer > 0.0 {
                similarity /= normalizer;
            }
        }

        Ok(similarity)
    }
}

impl Kernel for NGramKernel {
    fn compute(&self, _x: &[f64], _y: &[f64]) -> Result<f64> {
        // Placeholder - use compute_strings for string data
        Ok(0.0)
    }

    fn name(&self) -> &str {
        "NGram"
    }
}

/// Subsequence string kernel configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubsequenceKernelConfig {
    /// Maximum subsequence length
    pub max_length: usize,
    /// Decay factor for longer subsequences
    pub decay: f64,
}

impl SubsequenceKernelConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            max_length: 3,
            decay: 0.5,
        }
    }

    /// Set maximum length
    pub fn with_max_length(mut self, length: usize) -> Result<Self> {
        if length == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "max_length".to_string(),
                value: length.to_string(),
                reason: "max_length must be positive".to_string(),
            });
        }
        self.max_length = length;
        Ok(self)
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) {
            return Err(KernelError::InvalidParameter {
                parameter: "decay".to_string(),
                value: decay.to_string(),
                reason: "decay must be in [0, 1]".to_string(),
            });
        }
        self.decay = decay;
        Ok(self)
    }
}

impl Default for SubsequenceKernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Subsequence string kernel
///
/// Measures similarity by counting common non-contiguous subsequences.
pub struct SubsequenceKernel {
    config: SubsequenceKernelConfig,
}

impl SubsequenceKernel {
    /// Create a new subsequence kernel
    pub fn new(config: SubsequenceKernelConfig) -> Self {
        Self { config }
    }

    /// Compute similarity between two text strings
    pub fn compute_strings(&self, text1: &str, text2: &str) -> Result<f64> {
        let chars1: Vec<char> = text1.chars().collect();
        let chars2: Vec<char> = text2.chars().collect();

        let mut similarity = 0.0;

        // Use dynamic programming to count common subsequences
        for length in 1..=self.config.max_length.min(chars1.len()).min(chars2.len()) {
            let count = self.count_common_subsequences(&chars1, &chars2, length);
            similarity += count as f64 * self.config.decay.powi(length as i32);
        }

        Ok(similarity)
    }

    /// Count common subsequences of given length
    fn count_common_subsequences(&self, s1: &[char], s2: &[char], length: usize) -> usize {
        if length > s1.len() || length > s2.len() {
            return 0;
        }

        // Simplified counting - exact match of subsequences
        let subseqs1 = self.extract_subsequences(s1, length);
        let subseqs2 = self.extract_subsequences(s2, length);

        let mut count = 0;
        for subseq in &subseqs1 {
            if subseqs2.contains(subseq) {
                count += 1;
            }
        }

        count
    }

    /// Extract all subsequences of given length
    fn extract_subsequences(&self, chars: &[char], length: usize) -> Vec<Vec<char>> {
        let mut subsequences = Vec::new();
        self.generate_subsequences(chars, length, 0, Vec::new(), &mut subsequences);
        subsequences
    }

    /// Generate subsequences recursively
    #[allow(clippy::only_used_in_recursion)]
    fn generate_subsequences(
        &self,
        chars: &[char],
        remaining: usize,
        start: usize,
        current: Vec<char>,
        result: &mut Vec<Vec<char>>,
    ) {
        if remaining == 0 {
            result.push(current);
            return;
        }

        for i in start..chars.len() {
            let mut new_current = current.clone();
            new_current.push(chars[i]);
            self.generate_subsequences(chars, remaining - 1, i + 1, new_current, result);
        }
    }
}

impl Kernel for SubsequenceKernel {
    fn compute(&self, _x: &[f64], _y: &[f64]) -> Result<f64> {
        // Placeholder - use compute_strings for string data
        Ok(0.0)
    }

    fn name(&self) -> &str {
        "Subsequence"
    }
}

/// Edit distance kernel (exponential of negative edit distance)
///
/// K(s1, s2) = exp(-gamma * edit_distance(s1, s2))
pub struct EditDistanceKernel {
    /// Bandwidth parameter
    gamma: f64,
}

impl EditDistanceKernel {
    /// Create a new edit distance kernel
    pub fn new(gamma: f64) -> Result<Self> {
        if gamma <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "gamma".to_string(),
                value: gamma.to_string(),
                reason: "gamma must be positive".to_string(),
            });
        }

        Ok(Self { gamma })
    }

    /// Compute Levenshtein edit distance
    #[allow(clippy::needless_range_loop)]
    fn edit_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let m = chars1.len();
        let n = chars2.len();

        let mut dp = vec![vec![0; n + 1]; m + 1];

        // Initialize
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

                dp[i][j] = (dp[i - 1][j] + 1) // deletion
                    .min(dp[i][j - 1] + 1) // insertion
                    .min(dp[i - 1][j - 1] + cost); // substitution
            }
        }

        dp[m][n]
    }

    /// Compute similarity between two text strings
    pub fn compute_strings(&self, text1: &str, text2: &str) -> Result<f64> {
        let distance = self.edit_distance(text1, text2);
        let similarity = (-self.gamma * distance as f64).exp();
        Ok(similarity)
    }
}

impl Kernel for EditDistanceKernel {
    fn compute(&self, _x: &[f64], _y: &[f64]) -> Result<f64> {
        // Placeholder - use compute_strings for string data
        Ok(0.0)
    }

    fn name(&self) -> &str {
        "EditDistance"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_kernel() {
        let config = NGramKernelConfig::new(2).unwrap();
        let kernel = NGramKernel::new(config);

        let text1 = "hello";
        let text2 = "hallo";

        let sim = kernel.compute_strings(text1, text2).unwrap();
        assert!(sim > 0.0);
        assert!(sim <= 1.0);
    }

    #[test]
    fn test_ngram_identical_strings() {
        let config = NGramKernelConfig::new(2).unwrap();
        let kernel = NGramKernel::new(config);

        let text = "test";
        let sim = kernel.compute_strings(text, text).unwrap();

        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ngram_different_strings() {
        let config = NGramKernelConfig::new(2).unwrap();
        let kernel = NGramKernel::new(config);

        let text1 = "abc";
        let text2 = "xyz";

        let sim = kernel.compute_strings(text1, text2).unwrap();
        assert!(sim < 0.1); // Should be very low similarity
    }

    #[test]
    fn test_ngram_config_invalid_n() {
        let result = NGramKernelConfig::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_subsequence_kernel() {
        let config = SubsequenceKernelConfig::new();
        let kernel = SubsequenceKernel::new(config);

        let text1 = "abc";
        let text2 = "aec";

        let sim = kernel.compute_strings(text1, text2).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_subsequence_identical() {
        let config = SubsequenceKernelConfig::new();
        let kernel = SubsequenceKernel::new(config);

        let text = "test";
        let sim = kernel.compute_strings(text, text).unwrap();

        assert!(sim > 0.0);
    }

    #[test]
    fn test_subsequence_config() {
        let config = SubsequenceKernelConfig::new()
            .with_max_length(5)
            .unwrap()
            .with_decay(0.7)
            .unwrap();

        assert_eq!(config.max_length, 5);
        assert!((config.decay - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_subsequence_invalid_config() {
        let result = SubsequenceKernelConfig::new().with_max_length(0);
        assert!(result.is_err());

        let result = SubsequenceKernelConfig::new().with_decay(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_edit_distance_kernel() {
        let kernel = EditDistanceKernel::new(0.1).unwrap();

        let text1 = "kitten";
        let text2 = "sitting";

        let sim = kernel.compute_strings(text1, text2).unwrap();
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_edit_distance_identical() {
        let kernel = EditDistanceKernel::new(0.1).unwrap();

        let text = "test";
        let sim = kernel.compute_strings(text, text).unwrap();

        assert!((sim - 1.0).abs() < 1e-10); // exp(-0 * 0.1) = 1.0
    }

    #[test]
    fn test_edit_distance_computation() {
        let kernel = EditDistanceKernel::new(1.0).unwrap();

        assert_eq!(kernel.edit_distance("", ""), 0);
        assert_eq!(kernel.edit_distance("a", ""), 1);
        assert_eq!(kernel.edit_distance("", "a"), 1);
        assert_eq!(kernel.edit_distance("abc", "abc"), 0);
        assert_eq!(kernel.edit_distance("abc", "abd"), 1);
        assert_eq!(kernel.edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_edit_distance_invalid_gamma() {
        let result = EditDistanceKernel::new(-0.1);
        assert!(result.is_err());

        let result = EditDistanceKernel::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_trait() {
        let kernel = NGramKernel::new(NGramKernelConfig::new(2).unwrap());
        assert_eq!(kernel.name(), "NGram");

        let kernel = SubsequenceKernel::new(SubsequenceKernelConfig::new());
        assert_eq!(kernel.name(), "Subsequence");

        let kernel = EditDistanceKernel::new(0.1).unwrap();
        assert_eq!(kernel.name(), "EditDistance");
    }
}
