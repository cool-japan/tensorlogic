//! Batch management and data loading.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};
use std::collections::HashSet;

/// Configuration for batch processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Whether to shuffle data.
    pub shuffle: bool,
    /// Whether to drop last incomplete batch.
    pub drop_last: bool,
    /// Random seed for shuffling.
    pub seed: Option<u64>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            seed: None,
        }
    }
}

/// Iterator over batches of data.
pub struct BatchIterator {
    /// Configuration.
    config: BatchConfig,
    /// Total number of samples.
    num_samples: usize,
    /// Current batch index.
    current_batch: usize,
    /// Shuffled indices (if shuffle=true).
    indices: Vec<usize>,
}

impl BatchIterator {
    /// Create a new batch iterator.
    pub fn new(num_samples: usize, config: BatchConfig) -> Self {
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if config.shuffle {
            // Simple shuffle using seed if provided
            if let Some(seed) = config.seed {
                // Deterministic shuffle based on seed
                let mut rng_state = seed;
                for i in (1..indices.len()).rev() {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng_state % (i as u64 + 1)) as usize;
                    indices.swap(i, j);
                }
            } else {
                // Non-deterministic shuffle using simple algorithm
                use std::collections::hash_map::RandomState;
                use std::hash::BuildHasher;
                let hasher = RandomState::new();
                indices.sort_by_cached_key(|&i| hasher.hash_one(i));
            }
        }

        Self {
            config,
            num_samples,
            current_batch: 0,
            indices,
        }
    }

    /// Get the next batch indices.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current_batch * self.config.batch_size >= self.num_samples {
            return None;
        }

        let start = self.current_batch * self.config.batch_size;
        let end = (start + self.config.batch_size).min(self.num_samples);

        if self.config.drop_last && end - start < self.config.batch_size {
            return None;
        }

        self.current_batch += 1;
        Some(self.indices[start..end].to_vec())
    }

    /// Reset iterator to the beginning.
    pub fn reset(&mut self) {
        self.current_batch = 0;

        if self.config.shuffle {
            // Re-shuffle for next epoch
            if let Some(seed) = self.config.seed {
                let mut rng_state = seed.wrapping_add(self.current_batch as u64);
                for i in (1..self.indices.len()).rev() {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng_state % (i as u64 + 1)) as usize;
                    self.indices.swap(i, j);
                }
            } else {
                use std::collections::hash_map::RandomState;
                use std::hash::BuildHasher;
                let hasher = RandomState::new();
                self.indices
                    .sort_by_cached_key(|&i| hasher.hash_one((i, self.current_batch)));
            }
        }
    }

    /// Get total number of batches.
    pub fn num_batches(&self) -> usize {
        let total = self.num_samples.div_ceil(self.config.batch_size);
        if self.config.drop_last && !self.num_samples.is_multiple_of(self.config.batch_size) {
            total - 1
        } else {
            total
        }
    }
}

/// Data shuffler for randomizing training data.
pub struct DataShuffler {
    /// Random seed.
    #[allow(dead_code)]
    seed: Option<u64>,
    /// Internal state for random number generation.
    rng_state: u64,
}

impl DataShuffler {
    /// Create a new data shuffler.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            seed,
            rng_state: seed.unwrap_or(42),
        }
    }

    /// Shuffle indices.
    pub fn shuffle(&mut self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let j = (self.rng_state % (i as u64 + 1)) as usize;
            indices.swap(i, j);
        }
    }

    /// Generate a random permutation.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        self.shuffle(&mut indices);
        indices
    }
}

/// Extract batches from data arrays.
pub fn extract_batch(
    data: &ArrayView<f64, Ix2>,
    indices: &[usize],
) -> TrainResult<Array<f64, Ix2>> {
    let batch_size = indices.len();
    let num_features = data.ncols();
    let mut batch = Array::zeros((batch_size, num_features));

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= data.nrows() {
            return Err(TrainError::BatchError(format!(
                "Index {} out of bounds for data with {} rows",
                idx,
                data.nrows()
            )));
        }
        batch.row_mut(i).assign(&data.row(idx));
    }

    Ok(batch)
}

/// Stratified batch sampler for balanced class sampling.
#[allow(dead_code)]
pub struct StratifiedSampler {
    /// Class labels for each sample.
    labels: Vec<usize>,
    /// Indices for each class.
    class_indices: Vec<Vec<usize>>,
    /// Current position in each class.
    class_positions: Vec<usize>,
    /// Batch size.
    batch_size: usize,
    /// Random seed.
    seed: Option<u64>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler.
    #[allow(dead_code)]
    pub fn new(labels: Vec<usize>, batch_size: usize, seed: Option<u64>) -> TrainResult<Self> {
        if labels.is_empty() {
            return Err(TrainError::BatchError("Empty labels".to_string()));
        }

        // Find unique classes
        let unique_classes: HashSet<usize> = labels.iter().copied().collect();
        let num_classes = unique_classes.len();

        // Group indices by class
        let mut class_indices = vec![Vec::new(); num_classes];
        for (idx, &label) in labels.iter().enumerate() {
            class_indices[label].push(idx);
        }

        // Shuffle each class independently
        let mut shuffler = DataShuffler::new(seed);
        for class_idx in &mut class_indices {
            shuffler.shuffle(class_idx);
        }

        Ok(Self {
            labels,
            class_indices,
            class_positions: vec![0; num_classes],
            batch_size,
            seed,
        })
    }

    /// Get next stratified batch.
    #[allow(dead_code)]
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let num_classes = self.class_indices.len();
        let samples_per_class = self.batch_size / num_classes;

        let mut batch_indices = Vec::new();

        for class_id in 0..num_classes {
            let class_samples = &self.class_indices[class_id];
            let pos = self.class_positions[class_id];

            // Check if we have enough samples for this class
            if pos + samples_per_class > class_samples.len() {
                // Not enough samples for a complete stratified batch
                return None;
            }

            // Add samples from this class
            for i in 0..samples_per_class {
                batch_indices.push(class_samples[pos + i]);
            }

            self.class_positions[class_id] += samples_per_class;
        }

        if batch_indices.is_empty() {
            None
        } else {
            Some(batch_indices)
        }
    }

    /// Reset sampler.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.class_positions.fill(0);

        // Re-shuffle each class
        let mut shuffler = DataShuffler::new(self.seed);
        for class_idx in &mut self.class_indices {
            shuffler.shuffle(class_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_batch_iterator() {
        let config = BatchConfig {
            batch_size: 3,
            shuffle: false,
            drop_last: false,
            seed: Some(42),
        };
        let mut iter = BatchIterator::new(10, config);

        let batch1 = iter.next_batch().unwrap();
        assert_eq!(batch1.len(), 3);

        let batch2 = iter.next_batch().unwrap();
        assert_eq!(batch2.len(), 3);

        let batch3 = iter.next_batch().unwrap();
        assert_eq!(batch3.len(), 3);

        let batch4 = iter.next_batch().unwrap();
        assert_eq!(batch4.len(), 1); // Last batch with remaining samples

        assert!(iter.next_batch().is_none());
    }

    #[test]
    fn test_batch_iterator_drop_last() {
        let config = BatchConfig {
            batch_size: 3,
            shuffle: false,
            drop_last: true,
            seed: Some(42),
        };
        let mut iter = BatchIterator::new(10, config);

        let batch1 = iter.next_batch().unwrap();
        assert_eq!(batch1.len(), 3);

        let batch2 = iter.next_batch().unwrap();
        assert_eq!(batch2.len(), 3);

        let batch3 = iter.next_batch().unwrap();
        assert_eq!(batch3.len(), 3);

        assert!(iter.next_batch().is_none()); // Last incomplete batch is dropped
    }

    #[test]
    fn test_data_shuffler() {
        let mut shuffler = DataShuffler::new(Some(42));
        let mut indices = vec![0, 1, 2, 3, 4];
        let original = indices.clone();

        shuffler.shuffle(&mut indices);
        assert_ne!(indices, original); // Should be shuffled
        assert_eq!(indices.len(), original.len());
    }

    #[test]
    fn test_extract_batch() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let indices = vec![0, 2];

        let batch = extract_batch(&data.view(), &indices).unwrap();
        assert_eq!(batch.shape(), &[2, 2]);
        assert_eq!(batch[[0, 0]], 1.0);
        assert_eq!(batch[[0, 1]], 2.0);
        assert_eq!(batch[[1, 0]], 5.0);
        assert_eq!(batch[[1, 1]], 6.0);
    }

    #[test]
    fn test_stratified_sampler() {
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut sampler = StratifiedSampler::new(labels, 6, Some(42)).unwrap();

        let batch = sampler.next_batch().unwrap();
        assert_eq!(batch.len(), 6);

        // Count class distribution in batch
        let mut class_counts = vec![0; 3];
        for &idx in &batch {
            class_counts[sampler.labels[idx]] += 1;
        }

        // Each class should have 2 samples (6 / 3 classes)
        assert_eq!(class_counts, vec![2, 2, 2]);
    }
}
