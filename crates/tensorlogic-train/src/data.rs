//! Data loading and preprocessing utilities for training.
//!
//! This module provides tools for loading and preprocessing training data:
//! - CSV and JSON data loading
//! - Data normalization and standardization
//! - Train/validation/test splitting
//! - Data shuffling and sampling

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Dataset container for training data.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (samples x features).
    pub features: Array2<f64>,
    /// Target vector or matrix.
    pub targets: Array2<f64>,
    /// Feature names (if available).
    pub feature_names: Option<Vec<String>>,
    /// Target names (if available).
    pub target_names: Option<Vec<String>>,
}

impl Dataset {
    /// Create a new dataset.
    pub fn new(features: Array2<f64>, targets: Array2<f64>) -> Self {
        Self {
            features,
            targets,
            feature_names: None,
            target_names: None,
        }
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set target names.
    pub fn with_target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Get number of samples.
    pub fn num_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Get number of features.
    pub fn num_features(&self) -> usize {
        self.features.ncols()
    }

    /// Get number of targets.
    pub fn num_targets(&self) -> usize {
        self.targets.ncols()
    }

    /// Shuffle the dataset in place using Fisher-Yates algorithm.
    pub fn shuffle(&mut self, seed: u64) {
        let n = self.num_samples();
        if n <= 1 {
            return;
        }

        // Simple LCG for deterministic shuffling
        let mut rng_state = seed;
        let lcg_next = |state: &mut u64| -> usize {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as usize
        };

        for i in (1..n).rev() {
            let j = lcg_next(&mut rng_state) % (i + 1);
            // Swap rows
            for col in 0..self.features.ncols() {
                let tmp = self.features[[i, col]];
                self.features[[i, col]] = self.features[[j, col]];
                self.features[[j, col]] = tmp;
            }
            for col in 0..self.targets.ncols() {
                let tmp = self.targets[[i, col]];
                self.targets[[i, col]] = self.targets[[j, col]];
                self.targets[[j, col]] = tmp;
            }
        }
    }

    /// Split dataset into subsets.
    ///
    /// # Arguments
    /// * `ratios` - Ratios for each split (must sum to 1.0)
    ///
    /// # Returns
    /// Vector of datasets corresponding to each ratio
    pub fn split(&self, ratios: &[f64]) -> TrainResult<Vec<Dataset>> {
        let total: f64 = ratios.iter().sum();
        if (total - 1.0).abs() > 1e-6 {
            return Err(TrainError::ConfigError(format!(
                "Split ratios must sum to 1.0, got {}",
                total
            )));
        }

        let n = self.num_samples();
        let mut splits = Vec::new();
        let mut start = 0;

        for (i, &ratio) in ratios.iter().enumerate() {
            let end = if i == ratios.len() - 1 {
                n // Last split gets remaining samples
            } else {
                start + (n as f64 * ratio).round() as usize
            };

            let features = self.features.slice(s![start..end, ..]).to_owned();
            let targets = self.targets.slice(s![start..end, ..]).to_owned();

            let mut dataset = Dataset::new(features, targets);
            if let Some(ref names) = self.feature_names {
                dataset.feature_names = Some(names.clone());
            }
            if let Some(ref names) = self.target_names {
                dataset.target_names = Some(names.clone());
            }

            splits.push(dataset);
            start = end;
        }

        Ok(splits)
    }

    /// Split into train and test sets.
    pub fn train_test_split(&self, train_ratio: f64) -> TrainResult<(Dataset, Dataset)> {
        let splits = self.split(&[train_ratio, 1.0 - train_ratio])?;
        let mut iter = splits.into_iter();
        Ok((iter.next().unwrap(), iter.next().unwrap()))
    }

    /// Split into train, validation, and test sets.
    pub fn train_val_test_split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> TrainResult<(Dataset, Dataset, Dataset)> {
        let test_ratio = 1.0 - train_ratio - val_ratio;
        if test_ratio < 0.0 {
            return Err(TrainError::ConfigError(
                "Train and validation ratios exceed 1.0".to_string(),
            ));
        }
        let splits = self.split(&[train_ratio, val_ratio, test_ratio])?;
        let mut iter = splits.into_iter();
        Ok((
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        ))
    }

    /// Get a subset of the dataset by indices.
    pub fn subset(&self, indices: &[usize]) -> TrainResult<Dataset> {
        let n = self.num_samples();
        for &idx in indices {
            if idx >= n {
                return Err(TrainError::ConfigError(format!(
                    "Index {} out of bounds for dataset with {} samples",
                    idx, n
                )));
            }
        }

        let features = Array2::from_shape_fn((indices.len(), self.num_features()), |(i, j)| {
            self.features[[indices[i], j]]
        });
        let targets = Array2::from_shape_fn((indices.len(), self.num_targets()), |(i, j)| {
            self.targets[[indices[i], j]]
        });

        let mut dataset = Dataset::new(features, targets);
        dataset.feature_names = self.feature_names.clone();
        dataset.target_names = self.target_names.clone();

        Ok(dataset)
    }
}

/// CSV data loader.
#[derive(Debug, Clone)]
pub struct CsvLoader {
    /// Whether the CSV has a header row.
    pub has_header: bool,
    /// Delimiter character.
    pub delimiter: char,
    /// Indices of target columns (0-based).
    pub target_columns: Vec<usize>,
    /// Columns to skip.
    pub skip_columns: Vec<usize>,
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: ',',
            target_columns: vec![],
            skip_columns: vec![],
        }
    }
}

impl CsvLoader {
    /// Create a new CSV loader.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether CSV has a header.
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the delimiter character.
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set target column indices.
    pub fn with_target_columns(mut self, columns: Vec<usize>) -> Self {
        self.target_columns = columns;
        self
    }

    /// Set columns to skip.
    pub fn with_skip_columns(mut self, columns: Vec<usize>) -> Self {
        self.skip_columns = columns;
        self
    }

    /// Load data from a CSV file.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> TrainResult<Dataset> {
        let file = File::open(path.as_ref())
            .map_err(|e| TrainError::Other(format!("Failed to open CSV file: {}", e)))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut feature_names = None;
        let mut target_names = None;

        // Parse header if present
        if self.has_header {
            if let Some(Ok(header)) = lines.next() {
                let names: Vec<String> = header
                    .split(self.delimiter)
                    .map(|s| s.trim().to_string())
                    .collect();

                let mut feat_names = Vec::new();
                let mut targ_names = Vec::new();

                for (i, name) in names.into_iter().enumerate() {
                    if self.skip_columns.contains(&i) {
                        continue;
                    }
                    if self.target_columns.contains(&i) {
                        targ_names.push(name);
                    } else {
                        feat_names.push(name);
                    }
                }

                feature_names = Some(feat_names);
                target_names = Some(targ_names);
            }
        }

        // Parse data rows
        let mut features_data: Vec<Vec<f64>> = Vec::new();
        let mut targets_data: Vec<Vec<f64>> = Vec::new();

        for line_result in lines {
            let line = line_result
                .map_err(|e| TrainError::Other(format!("Failed to read CSV line: {}", e)))?;

            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split(self.delimiter).collect();
            let mut row_features = Vec::new();
            let mut row_targets = Vec::new();

            for (i, value) in values.iter().enumerate() {
                if self.skip_columns.contains(&i) {
                    continue;
                }

                let parsed: f64 = value.trim().parse().map_err(|e| {
                    TrainError::Other(format!("Failed to parse value '{}': {}", value, e))
                })?;

                if self.target_columns.contains(&i) {
                    row_targets.push(parsed);
                } else {
                    row_features.push(parsed);
                }
            }

            features_data.push(row_features);
            targets_data.push(row_targets);
        }

        if features_data.is_empty() {
            return Err(TrainError::Other("CSV file is empty".to_string()));
        }

        let n_samples = features_data.len();
        let n_features = features_data[0].len();
        let n_targets = if targets_data[0].is_empty() {
            0
        } else {
            targets_data[0].len()
        };

        // Convert to arrays
        let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| features_data[i][j]);

        let targets = if n_targets > 0 {
            Array2::from_shape_fn((n_samples, n_targets), |(i, j)| targets_data[i][j])
        } else {
            Array2::zeros((n_samples, 1))
        };

        let mut dataset = Dataset::new(features, targets);
        dataset.feature_names = feature_names;
        dataset.target_names = target_names;

        Ok(dataset)
    }
}

/// Data preprocessor for normalization and standardization.
#[derive(Debug, Clone)]
pub struct DataPreprocessor {
    /// Preprocessing method.
    method: PreprocessingMethod,
    /// Fitted parameters (mean, std, min, max).
    params: Option<PreprocessingParams>,
}

/// Preprocessing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessingMethod {
    /// Standardization (zero mean, unit variance).
    Standardize,
    /// Min-max normalization to [0, 1].
    MinMaxNormalize,
    /// Min-max scaling to custom range.
    MinMaxScale { min: i32, max: i32 },
    /// No preprocessing.
    None,
}

/// Fitted preprocessing parameters.
#[derive(Debug, Clone)]
struct PreprocessingParams {
    means: Array1<f64>,
    stds: Array1<f64>,
    mins: Array1<f64>,
    maxs: Array1<f64>,
}

impl DataPreprocessor {
    /// Create a new preprocessor with standardization.
    pub fn standardize() -> Self {
        Self {
            method: PreprocessingMethod::Standardize,
            params: None,
        }
    }

    /// Create a new preprocessor with min-max normalization.
    pub fn min_max_normalize() -> Self {
        Self {
            method: PreprocessingMethod::MinMaxNormalize,
            params: None,
        }
    }

    /// Create a new preprocessor with custom min-max scaling.
    pub fn min_max_scale(min: i32, max: i32) -> Self {
        Self {
            method: PreprocessingMethod::MinMaxScale { min, max },
            params: None,
        }
    }

    /// Create a preprocessor that does nothing.
    pub fn none() -> Self {
        Self {
            method: PreprocessingMethod::None,
            params: None,
        }
    }

    /// Fit the preprocessor to data.
    pub fn fit(&mut self, data: &Array2<f64>) -> &mut Self {
        let n_features = data.ncols();

        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);
        let mut mins = Array1::from_elem(n_features, f64::INFINITY);
        let mut maxs = Array1::from_elem(n_features, f64::NEG_INFINITY);

        for j in 0..n_features {
            let col = data.column(j);
            let n = col.len() as f64;

            // Compute mean
            let mean: f64 = col.iter().sum::<f64>() / n;
            means[j] = mean;

            // Compute std
            let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            stds[j] = variance.sqrt().max(1e-8); // Avoid division by zero

            // Compute min/max
            for &x in col.iter() {
                if x < mins[j] {
                    mins[j] = x;
                }
                if x > maxs[j] {
                    maxs[j] = x;
                }
            }
        }

        self.params = Some(PreprocessingParams {
            means,
            stds,
            mins,
            maxs,
        });

        self
    }

    /// Transform data using fitted parameters.
    pub fn transform(&self, data: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let params = self.params.as_ref().ok_or_else(|| {
            TrainError::Other("Preprocessor not fitted. Call fit() first.".to_string())
        })?;

        let mut result = data.clone();

        match self.method {
            PreprocessingMethod::Standardize => {
                for j in 0..data.ncols() {
                    for i in 0..data.nrows() {
                        result[[i, j]] = (data[[i, j]] - params.means[j]) / params.stds[j];
                    }
                }
            }
            PreprocessingMethod::MinMaxNormalize => {
                for j in 0..data.ncols() {
                    let range = (params.maxs[j] - params.mins[j]).max(1e-8);
                    for i in 0..data.nrows() {
                        result[[i, j]] = (data[[i, j]] - params.mins[j]) / range;
                    }
                }
            }
            PreprocessingMethod::MinMaxScale { min, max } => {
                let target_range = (max - min) as f64;
                for j in 0..data.ncols() {
                    let range = (params.maxs[j] - params.mins[j]).max(1e-8);
                    for i in 0..data.nrows() {
                        let normalized = (data[[i, j]] - params.mins[j]) / range;
                        result[[i, j]] = normalized * target_range + min as f64;
                    }
                }
            }
            PreprocessingMethod::None => {}
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> TrainResult<Array2<f64>> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform to original scale.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let params = self.params.as_ref().ok_or_else(|| {
            TrainError::Other("Preprocessor not fitted. Call fit() first.".to_string())
        })?;

        let mut result = data.clone();

        match self.method {
            PreprocessingMethod::Standardize => {
                for j in 0..data.ncols() {
                    for i in 0..data.nrows() {
                        result[[i, j]] = data[[i, j]] * params.stds[j] + params.means[j];
                    }
                }
            }
            PreprocessingMethod::MinMaxNormalize => {
                for j in 0..data.ncols() {
                    let range = params.maxs[j] - params.mins[j];
                    for i in 0..data.nrows() {
                        result[[i, j]] = data[[i, j]] * range + params.mins[j];
                    }
                }
            }
            PreprocessingMethod::MinMaxScale { min, max } => {
                let target_range = (max - min) as f64;
                for j in 0..data.ncols() {
                    let range = params.maxs[j] - params.mins[j];
                    for i in 0..data.nrows() {
                        let normalized = (data[[i, j]] - min as f64) / target_range;
                        result[[i, j]] = normalized * range + params.mins[j];
                    }
                }
            }
            PreprocessingMethod::None => {}
        }

        Ok(result)
    }

    /// Check if the preprocessor is fitted.
    pub fn is_fitted(&self) -> bool {
        self.params.is_some()
    }

    /// Get the preprocessing method.
    pub fn method(&self) -> PreprocessingMethod {
        self.method
    }
}

/// One-hot encoder for categorical data.
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    /// Mapping from category to index for each column.
    categories: HashMap<usize, HashMap<String, usize>>,
    /// Number of categories per column.
    n_categories: HashMap<usize, usize>,
}

impl OneHotEncoder {
    /// Create a new one-hot encoder.
    pub fn new() -> Self {
        Self {
            categories: HashMap::new(),
            n_categories: HashMap::new(),
        }
    }

    /// Fit the encoder to categorical data.
    ///
    /// # Arguments
    /// * `data` - Vector of (column_index, values) pairs
    pub fn fit(&mut self, data: &[(usize, Vec<String>)]) -> &mut Self {
        for (col_idx, values) in data {
            let mut categories = HashMap::new();
            let mut unique_values: Vec<&String> = values.iter().collect();
            unique_values.sort();
            unique_values.dedup();

            for (i, value) in unique_values.into_iter().enumerate() {
                categories.insert(value.clone(), i);
            }

            self.n_categories.insert(*col_idx, categories.len());
            self.categories.insert(*col_idx, categories);
        }

        self
    }

    /// Transform categorical column to one-hot encoded array.
    pub fn transform(&self, col_idx: usize, values: &[String]) -> TrainResult<Array2<f64>> {
        let categories = self
            .categories
            .get(&col_idx)
            .ok_or_else(|| TrainError::Other(format!("Column {} not fitted", col_idx)))?;

        let n_samples = values.len();
        let n_cats = *self.n_categories.get(&col_idx).unwrap();

        let mut result = Array2::zeros((n_samples, n_cats));

        for (i, value) in values.iter().enumerate() {
            if let Some(&idx) = categories.get(value) {
                result[[i, idx]] = 1.0;
            } else {
                return Err(TrainError::Other(format!(
                    "Unknown category '{}' for column {}",
                    value, col_idx
                )));
            }
        }

        Ok(result)
    }

    /// Get number of categories for a column.
    pub fn num_categories(&self, col_idx: usize) -> Option<usize> {
        self.n_categories.get(&col_idx).copied()
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Label encoder for converting string labels to integers.
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    /// Mapping from label to integer.
    label_to_int: HashMap<String, usize>,
    /// Mapping from integer to label.
    int_to_label: Vec<String>,
}

impl LabelEncoder {
    /// Create a new label encoder.
    pub fn new() -> Self {
        Self {
            label_to_int: HashMap::new(),
            int_to_label: Vec::new(),
        }
    }

    /// Fit the encoder to labels.
    pub fn fit(&mut self, labels: &[String]) -> &mut Self {
        let mut unique: Vec<&String> = labels.iter().collect();
        unique.sort();
        unique.dedup();

        self.label_to_int.clear();
        self.int_to_label.clear();

        for (i, label) in unique.into_iter().enumerate() {
            self.label_to_int.insert(label.clone(), i);
            self.int_to_label.push(label.clone());
        }

        self
    }

    /// Transform labels to integers.
    pub fn transform(&self, labels: &[String]) -> TrainResult<Array1<usize>> {
        let mut result = Array1::zeros(labels.len());

        for (i, label) in labels.iter().enumerate() {
            result[i] = *self
                .label_to_int
                .get(label)
                .ok_or_else(|| TrainError::Other(format!("Unknown label: {}", label)))?;
        }

        Ok(result)
    }

    /// Inverse transform integers to labels.
    pub fn inverse_transform(&self, indices: &Array1<usize>) -> TrainResult<Vec<String>> {
        let mut result = Vec::with_capacity(indices.len());

        for &idx in indices.iter() {
            if idx >= self.int_to_label.len() {
                return Err(TrainError::Other(format!(
                    "Index {} out of bounds for {} classes",
                    idx,
                    self.int_to_label.len()
                )));
            }
            result.push(self.int_to_label[idx].clone());
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, labels: &[String]) -> TrainResult<Array1<usize>> {
        self.fit(labels);
        self.transform(labels)
    }

    /// Get number of classes.
    pub fn num_classes(&self) -> usize {
        self.int_to_label.len()
    }

    /// Get class labels.
    pub fn classes(&self) -> &[String] {
        &self.int_to_label
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 0.0]).unwrap();

        let dataset = Dataset::new(features, targets);

        assert_eq!(dataset.num_samples(), 3);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.num_targets(), 1);
    }

    #[test]
    fn test_dataset_split() {
        let features = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f64);
        let targets = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);

        let dataset = Dataset::new(features, targets);
        let splits = dataset.split(&[0.6, 0.2, 0.2]).unwrap();

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].num_samples(), 6);
        assert_eq!(splits[1].num_samples(), 2);
        assert_eq!(splits[2].num_samples(), 2);
    }

    #[test]
    fn test_train_test_split() {
        let features = Array2::from_shape_fn((100, 4), |(i, j)| (i * 4 + j) as f64);
        let targets = Array2::from_shape_fn((100, 1), |(i, _)| (i % 2) as f64);

        let dataset = Dataset::new(features, targets);
        let (train, test) = dataset.train_test_split(0.8).unwrap();

        assert_eq!(train.num_samples(), 80);
        assert_eq!(test.num_samples(), 20);
    }

    #[test]
    fn test_dataset_shuffle() {
        let features = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f64);
        let targets = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);

        let mut dataset = Dataset::new(features.clone(), targets);
        dataset.shuffle(42);

        // After shuffle, data should be different
        let mut different = false;
        for i in 0..10 {
            if dataset.features[[i, 0]] != features[[i, 0]] {
                different = true;
                break;
            }
        }
        assert!(different);
    }

    #[test]
    fn test_dataset_subset() {
        let features = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f64);
        let targets = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);

        let dataset = Dataset::new(features, targets);
        let subset = dataset.subset(&[0, 2, 4]).unwrap();

        assert_eq!(subset.num_samples(), 3);
        assert_eq!(subset.features[[0, 0]], 0.0);
        assert_eq!(subset.features[[1, 0]], 4.0);
        assert_eq!(subset.features[[2, 0]], 8.0);
    }

    #[test]
    fn test_preprocessor_standardize() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let mut preprocessor = DataPreprocessor::standardize();
        let transformed = preprocessor.fit_transform(&data).unwrap();

        // Check that mean is approximately 0
        let col0_mean: f64 = transformed.column(0).iter().sum::<f64>() / 4.0;
        let col1_mean: f64 = transformed.column(1).iter().sum::<f64>() / 4.0;

        assert!(col0_mean.abs() < 1e-10);
        assert!(col1_mean.abs() < 1e-10);

        // Check inverse transform
        let recovered = preprocessor.inverse_transform(&transformed).unwrap();
        for i in 0..4 {
            for j in 0..2 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_preprocessor_min_max() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 10.0, 5.0, 20.0, 10.0, 30.0, 15.0, 40.0])
                .unwrap();

        let mut preprocessor = DataPreprocessor::min_max_normalize();
        let transformed = preprocessor.fit_transform(&data).unwrap();

        // Check that values are in [0, 1]
        for &val in transformed.iter() {
            assert!((0.0..=1.0).contains(&val));
        }

        // Check specific values
        assert!((transformed[[0, 0]] - 0.0).abs() < 1e-10); // min
        assert!((transformed[[3, 0]] - 1.0).abs() < 1e-10); // max
    }

    #[test]
    fn test_label_encoder() {
        let labels = vec![
            "cat".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "bird".to_string(),
        ];

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(&labels).unwrap();

        assert_eq!(encoder.num_classes(), 3);
        assert_eq!(encoded.len(), 4);

        // Same labels should have same encoding
        assert_eq!(encoded[0], encoded[2]);

        // Test inverse transform
        let decoded = encoder.inverse_transform(&encoded).unwrap();
        assert_eq!(decoded, labels);
    }

    #[test]
    fn test_one_hot_encoder() {
        let values = vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
            "red".to_string(),
        ];

        let mut encoder = OneHotEncoder::new();
        encoder.fit(&[(0, values.clone())]);

        let encoded = encoder.transform(0, &values).unwrap();

        assert_eq!(encoded.nrows(), 4);
        assert_eq!(encoded.ncols(), 3);

        // Each row should sum to 1
        for i in 0..4 {
            let row_sum: f64 = encoded.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_csv_loader_builder() {
        let loader = CsvLoader::new()
            .with_header(true)
            .with_delimiter(',')
            .with_target_columns(vec![3]);

        assert!(loader.has_header);
        assert_eq!(loader.delimiter, ',');
        assert_eq!(loader.target_columns, vec![3]);
    }

    #[test]
    fn test_invalid_split_ratios() {
        let features = Array2::zeros((10, 2));
        let targets = Array2::zeros((10, 1));
        let dataset = Dataset::new(features, targets);

        // Ratios don't sum to 1
        let result = dataset.split(&[0.5, 0.3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocessor_not_fitted() {
        let data = Array2::zeros((4, 2));
        let preprocessor = DataPreprocessor::standardize();

        let result = preprocessor.transform(&data);
        assert!(result.is_err());
    }
}
