//! Provenance tracking for kernel computations
//!
//! This module provides comprehensive tracking of kernel operations for debugging,
//! auditing, and reproducibility. It records metadata about each computation including
//! input shapes, parameters, execution time, and results.
//!
//! # Features
//!
//! - **Automatic Tracking**: Wrap any kernel with `ProvenanceKernel` for transparent tracking
//! - **Rich Metadata**: Records timestamps, computation time, input/output dimensions
//! - **Query Interface**: Search and filter provenance records by various criteria
//! - **Export Formats**: JSON serialization for analysis and archival
//! - **Memory Efficient**: Configurable history limits and sampling strategies
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{
//!     LinearKernel, Kernel,
//!     provenance::{ProvenanceKernel, ProvenanceTracker}
//! };
//!
//! // Create kernel with provenance tracking
//! let base_kernel = LinearKernel::new();
//! let tracker = ProvenanceTracker::new();
//! let kernel = ProvenanceKernel::new(Box::new(base_kernel), tracker.clone());
//!
//! // Computations are automatically tracked
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let result = kernel.compute(&x, &y).unwrap();
//!
//! // Query provenance history
//! let records = tracker.get_all_records();
//! println!("Tracked {} computations", records.len());
//!
//! // Analyze computation patterns
//! let avg_time = tracker.average_computation_time();
//! println!("Average computation time: {:?}", avg_time);
//! ```

use crate::error::{KernelError, Result};
use crate::types::Kernel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Unique identifier for a provenance record
pub type ProvenanceId = String;

/// A record of a single kernel computation with full metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Unique identifier for this computation
    pub id: ProvenanceId,

    /// Timestamp when computation started
    pub timestamp: SystemTime,

    /// Name of the kernel used
    pub kernel_name: String,

    /// Kernel parameters as key-value pairs
    pub kernel_params: HashMap<String, String>,

    /// Dimension of input vectors
    pub input_dimension: usize,

    /// Number of samples processed (1 for pairwise, n for matrix)
    pub num_samples: usize,

    /// Result of the computation
    pub result: ComputationResult,

    /// Time taken for computation
    pub computation_time: Duration,

    /// Optional tags for categorization
    pub tags: Vec<String>,

    /// Optional notes or metadata
    pub notes: Option<String>,
}

/// Result of a kernel computation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ComputationResult {
    /// Single pairwise similarity value
    Pairwise(f64),

    /// Kernel matrix (n×n)
    Matrix {
        dimension: usize,
        trace: f64,          // Sum of diagonal elements
        frobenius_norm: f64, // Frobenius norm of the matrix
    },

    /// Error during computation
    Error { message: String },
}

impl ProvenanceRecord {
    /// Create a new provenance record
    pub fn new(
        kernel_name: String,
        kernel_params: HashMap<String, String>,
        input_dimension: usize,
        num_samples: usize,
    ) -> Self {
        Self {
            id: Self::generate_id(),
            timestamp: SystemTime::now(),
            kernel_name,
            kernel_params,
            input_dimension,
            num_samples,
            result: ComputationResult::Pairwise(0.0),
            computation_time: Duration::from_secs(0),
            tags: Vec::new(),
            notes: None,
        }
    }

    /// Generate a unique ID for this record
    fn generate_id() -> ProvenanceId {
        use std::time::UNIX_EPOCH;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("prov_{}", timestamp)
    }

    /// Add a tag to this record
    pub fn add_tag(&mut self, tag: String) {
        self.tags.push(tag);
    }

    /// Add a note to this record
    pub fn add_note(&mut self, note: String) {
        self.notes = Some(note);
    }

    /// Check if computation was successful
    pub fn is_success(&self) -> bool {
        !matches!(self.result, ComputationResult::Error { .. })
    }
}

/// Configuration for provenance tracking
#[derive(Clone, Debug)]
pub struct ProvenanceConfig {
    /// Maximum number of records to keep (None = unlimited)
    pub max_records: Option<usize>,

    /// Whether to track pairwise computations
    pub track_pairwise: bool,

    /// Whether to track matrix computations
    pub track_matrix: bool,

    /// Sample rate (1.0 = track all, 0.5 = track 50%, etc.)
    pub sample_rate: f64,

    /// Whether to include detailed timing information
    pub include_timing: bool,
}

impl Default for ProvenanceConfig {
    fn default() -> Self {
        Self {
            max_records: Some(1000),
            track_pairwise: true,
            track_matrix: true,
            sample_rate: 1.0,
            include_timing: true,
        }
    }
}

impl ProvenanceConfig {
    /// Create a new configuration with all tracking enabled
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of records to keep
    pub fn with_max_records(mut self, max: usize) -> Self {
        self.max_records = Some(max);
        self
    }

    /// Enable unlimited record storage
    pub fn with_unlimited_records(mut self) -> Self {
        self.max_records = None;
        self
    }

    /// Set sample rate (0.0 to 1.0)
    pub fn with_sample_rate(mut self, rate: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(KernelError::InvalidParameter {
                parameter: "sample_rate".to_string(),
                value: rate.to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.sample_rate = rate;
        Ok(self)
    }

    /// Enable/disable timing information
    pub fn with_timing(mut self, enable: bool) -> Self {
        self.include_timing = enable;
        self
    }
}

/// Thread-safe provenance tracker
#[derive(Clone)]
pub struct ProvenanceTracker {
    records: Arc<Mutex<Vec<ProvenanceRecord>>>,
    config: Arc<ProvenanceConfig>,
    counter: Arc<Mutex<usize>>,
}

impl ProvenanceTracker {
    /// Create a new provenance tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(ProvenanceConfig::default())
    }

    /// Create a new provenance tracker with custom configuration
    pub fn with_config(config: ProvenanceConfig) -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(config),
            counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Add a record to the tracker
    pub fn add_record(&self, record: ProvenanceRecord) {
        // Check if we should sample this record
        if self.config.sample_rate < 1.0 {
            let mut counter = self.counter.lock().unwrap();
            *counter += 1;
            let sample_every = (1.0 / self.config.sample_rate) as usize;
            if !(*counter).is_multiple_of(sample_every) {
                return;
            }
        }

        let mut records = self.records.lock().unwrap();
        records.push(record);

        // Enforce max_records limit
        if let Some(max) = self.config.max_records {
            if records.len() > max {
                records.remove(0); // Remove oldest
            }
        }
    }

    /// Get all records
    pub fn get_all_records(&self) -> Vec<ProvenanceRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Get records filtered by kernel name
    pub fn get_records_by_kernel(&self, kernel_name: &str) -> Vec<ProvenanceRecord> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.kernel_name == kernel_name)
            .cloned()
            .collect()
    }

    /// Get records filtered by tag
    pub fn get_records_by_tag(&self, tag: &str) -> Vec<ProvenanceRecord> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.tags.contains(&tag.to_string()))
            .cloned()
            .collect()
    }

    /// Get number of tracked records
    pub fn count(&self) -> usize {
        self.records.lock().unwrap().len()
    }

    /// Clear all records
    pub fn clear(&self) {
        self.records.lock().unwrap().clear();
    }

    /// Get average computation time across all records
    pub fn average_computation_time(&self) -> Option<Duration> {
        let records = self.records.lock().unwrap();
        if records.is_empty() {
            return None;
        }

        let total: Duration = records.iter().map(|r| r.computation_time).sum();
        Some(total / records.len() as u32)
    }

    /// Get statistics about tracked computations
    pub fn statistics(&self) -> ProvenanceStatistics {
        let records = self.records.lock().unwrap();

        let total_computations = records.len();
        let successful_computations = records.iter().filter(|r| r.is_success()).count();
        let failed_computations = total_computations - successful_computations;

        let mut kernel_counts: HashMap<String, usize> = HashMap::new();
        for record in records.iter() {
            *kernel_counts.entry(record.kernel_name.clone()).or_insert(0) += 1;
        }

        let total_time: Duration = records.iter().map(|r| r.computation_time).sum();
        let avg_time = if !records.is_empty() {
            Some(total_time / records.len() as u32)
        } else {
            None
        };

        ProvenanceStatistics {
            total_computations,
            successful_computations,
            failed_computations,
            kernel_counts,
            total_computation_time: total_time,
            average_computation_time: avg_time,
        }
    }

    /// Export records to JSON
    pub fn to_json(&self) -> Result<String> {
        let records = self.records.lock().unwrap();
        serde_json::to_string_pretty(&*records).map_err(|e| {
            KernelError::ComputationError(format!("Failed to serialize provenance records: {}", e))
        })
    }

    /// Import records from JSON
    pub fn from_json(&self, json: &str) -> Result<()> {
        let imported: Vec<ProvenanceRecord> = serde_json::from_str(json).map_err(|e| {
            KernelError::ComputationError(format!(
                "Failed to deserialize provenance records: {}",
                e
            ))
        })?;

        let mut records = self.records.lock().unwrap();
        records.extend(imported);
        Ok(())
    }
}

impl Default for ProvenanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about tracked computations
#[derive(Clone, Debug)]
pub struct ProvenanceStatistics {
    pub total_computations: usize,
    pub successful_computations: usize,
    pub failed_computations: usize,
    pub kernel_counts: HashMap<String, usize>,
    pub total_computation_time: Duration,
    pub average_computation_time: Option<Duration>,
}

/// A kernel wrapper that automatically tracks provenance
pub struct ProvenanceKernel {
    base_kernel: Box<dyn Kernel>,
    tracker: ProvenanceTracker,
    tags: Vec<String>,
}

impl ProvenanceKernel {
    /// Create a new provenance-tracking kernel
    pub fn new(base_kernel: Box<dyn Kernel>, tracker: ProvenanceTracker) -> Self {
        Self {
            base_kernel,
            tracker,
            tags: Vec::new(),
        }
    }

    /// Add a tag to all future computations
    pub fn add_tag(&mut self, tag: String) {
        self.tags.push(tag);
    }

    /// Get the underlying tracker
    pub fn tracker(&self) -> &ProvenanceTracker {
        &self.tracker
    }

    /// Create a provenance record for a computation
    fn create_record(&self, input_dimension: usize, num_samples: usize) -> ProvenanceRecord {
        let mut record = ProvenanceRecord::new(
            self.base_kernel.name().to_string(),
            self.get_kernel_params(),
            input_dimension,
            num_samples,
        );

        for tag in &self.tags {
            record.add_tag(tag.clone());
        }

        record
    }

    /// Extract kernel parameters as string map
    fn get_kernel_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "kernel_type".to_string(),
            self.base_kernel.name().to_string(),
        );
        // Add more parameters based on kernel type if needed
        params
    }
}

impl Kernel for ProvenanceKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let start = SystemTime::now();
        let mut record = self.create_record(x.len(), 1);

        let result = self.base_kernel.compute(x, y);

        let computation_time = start.elapsed().unwrap_or(Duration::from_secs(0));
        record.computation_time = computation_time;

        match result {
            Ok(value) => {
                record.result = ComputationResult::Pairwise(value);
                self.tracker.add_record(record);
                Ok(value)
            }
            Err(e) => {
                record.result = ComputationResult::Error {
                    message: e.to_string(),
                };
                self.tracker.add_record(record);
                Err(e)
            }
        }
    }

    fn compute_matrix(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let start = SystemTime::now();
        let mut record = self.create_record(data[0].len(), data.len());

        let result = self.base_kernel.compute_matrix(data);

        let computation_time = start.elapsed().unwrap_or(Duration::from_secs(0));
        record.computation_time = computation_time;

        match result {
            Ok(ref matrix) => {
                // Compute statistics
                let trace: f64 = (0..matrix.len()).map(|i| matrix[i][i]).sum();
                let frobenius_norm: f64 = matrix
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt();

                record.result = ComputationResult::Matrix {
                    dimension: matrix.len(),
                    trace,
                    frobenius_norm,
                };
                self.tracker.add_record(record);
                Ok(matrix.clone())
            }
            Err(e) => {
                record.result = ComputationResult::Error {
                    message: e.to_string(),
                };
                self.tracker.add_record(record);
                Err(e)
            }
        }
    }

    fn name(&self) -> &str {
        self.base_kernel.name()
    }

    fn is_psd(&self) -> bool {
        self.base_kernel.is_psd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernel::LinearKernel;

    #[test]
    fn test_provenance_record_creation() {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), "0.5".to_string());

        let record = ProvenanceRecord::new("RBF".to_string(), params, 10, 5);

        assert_eq!(record.kernel_name, "RBF");
        assert_eq!(record.input_dimension, 10);
        assert_eq!(record.num_samples, 5);
        assert!(!record.id.is_empty());
    }

    #[test]
    fn test_provenance_record_tags() {
        let mut record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 5, 1);

        record.add_tag("experiment1".to_string());
        record.add_tag("baseline".to_string());

        assert_eq!(record.tags.len(), 2);
        assert!(record.tags.contains(&"experiment1".to_string()));
    }

    #[test]
    fn test_provenance_config() {
        let config = ProvenanceConfig::new()
            .with_max_records(500)
            .with_sample_rate(0.5)
            .unwrap()
            .with_timing(true);

        assert_eq!(config.max_records, Some(500));
        assert!((config.sample_rate - 0.5).abs() < 1e-10);
        assert!(config.include_timing);
    }

    #[test]
    fn test_provenance_config_invalid_sample_rate() {
        let result = ProvenanceConfig::new().with_sample_rate(1.5);
        assert!(result.is_err());

        let result = ProvenanceConfig::new().with_sample_rate(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_provenance_tracker_basic() {
        let tracker = ProvenanceTracker::new();

        let record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);

        tracker.add_record(record.clone());

        assert_eq!(tracker.count(), 1);

        let records = tracker.get_all_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kernel_name, "Linear");
    }

    #[test]
    fn test_provenance_tracker_max_records() {
        let config = ProvenanceConfig::new().with_max_records(3);
        let tracker = ProvenanceTracker::with_config(config);

        for i in 0..5 {
            let record = ProvenanceRecord::new(format!("Kernel{}", i), HashMap::new(), 10, 1);
            tracker.add_record(record);
        }

        // Should only keep last 3 records
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_provenance_tracker_filter_by_kernel() {
        let tracker = ProvenanceTracker::new();

        for _ in 0..3 {
            let record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
            tracker.add_record(record);
        }

        for _ in 0..2 {
            let record = ProvenanceRecord::new("RBF".to_string(), HashMap::new(), 10, 1);
            tracker.add_record(record);
        }

        let linear_records = tracker.get_records_by_kernel("Linear");
        assert_eq!(linear_records.len(), 3);

        let rbf_records = tracker.get_records_by_kernel("RBF");
        assert_eq!(rbf_records.len(), 2);
    }

    #[test]
    fn test_provenance_tracker_filter_by_tag() {
        let tracker = ProvenanceTracker::new();

        let mut record1 = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
        record1.add_tag("experiment1".to_string());
        tracker.add_record(record1);

        let mut record2 = ProvenanceRecord::new("RBF".to_string(), HashMap::new(), 10, 1);
        record2.add_tag("experiment1".to_string());
        record2.add_tag("baseline".to_string());
        tracker.add_record(record2);

        let exp1_records = tracker.get_records_by_tag("experiment1");
        assert_eq!(exp1_records.len(), 2);

        let baseline_records = tracker.get_records_by_tag("baseline");
        assert_eq!(baseline_records.len(), 1);
    }

    #[test]
    fn test_provenance_tracker_statistics() {
        let tracker = ProvenanceTracker::new();

        for _ in 0..5 {
            let mut record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
            record.result = ComputationResult::Pairwise(1.0);
            record.computation_time = Duration::from_millis(10);
            tracker.add_record(record);
        }

        let stats = tracker.statistics();
        assert_eq!(stats.total_computations, 5);
        assert_eq!(stats.successful_computations, 5);
        assert_eq!(stats.failed_computations, 0);
        assert_eq!(stats.kernel_counts.get("Linear"), Some(&5));
    }

    #[test]
    fn test_provenance_tracker_clear() {
        let tracker = ProvenanceTracker::new();

        let record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
        tracker.add_record(record);

        assert_eq!(tracker.count(), 1);

        tracker.clear();
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_provenance_kernel_pairwise() {
        let base = Box::new(LinearKernel::new());
        let tracker = ProvenanceTracker::new();
        let kernel = ProvenanceKernel::new(base, tracker.clone());

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 32.0).abs() < 1e-10);

        assert_eq!(tracker.count(), 1);

        let records = tracker.get_all_records();
        assert_eq!(records[0].kernel_name, "Linear");
        assert_eq!(records[0].input_dimension, 3);
        assert_eq!(records[0].num_samples, 1);

        match records[0].result {
            ComputationResult::Pairwise(v) => assert!((v - 32.0).abs() < 1e-10),
            _ => panic!("Expected Pairwise result"),
        }
    }

    #[test]
    fn test_provenance_kernel_matrix() {
        let base = Box::new(LinearKernel::new());
        let tracker = ProvenanceTracker::new();
        let kernel = ProvenanceKernel::new(base, tracker.clone());

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let matrix = kernel.compute_matrix(&data).unwrap();
        assert_eq!(matrix.len(), 3);

        assert_eq!(tracker.count(), 1);

        let records = tracker.get_all_records();
        match &records[0].result {
            ComputationResult::Matrix { dimension, .. } => {
                assert_eq!(*dimension, 3);
            }
            _ => panic!("Expected Matrix result"),
        }
    }

    #[test]
    fn test_provenance_kernel_with_tags() {
        let base = Box::new(LinearKernel::new());
        let tracker = ProvenanceTracker::new();
        let mut kernel = ProvenanceKernel::new(base, tracker.clone());

        kernel.add_tag("experiment1".to_string());
        kernel.add_tag("baseline".to_string());

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        kernel.compute(&x, &y).unwrap();

        let records = tracker.get_all_records();
        assert_eq!(records[0].tags.len(), 2);
        assert!(records[0].tags.contains(&"experiment1".to_string()));
        assert!(records[0].tags.contains(&"baseline".to_string()));
    }

    #[test]
    fn test_provenance_json_serialization() {
        let tracker = ProvenanceTracker::new();

        let record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
        tracker.add_record(record);

        let json = tracker.to_json().unwrap();
        assert!(!json.is_empty());

        // Test deserialization
        let tracker2 = ProvenanceTracker::new();
        tracker2.from_json(&json).unwrap();

        assert_eq!(tracker2.count(), 1);
    }

    #[test]
    fn test_provenance_average_computation_time() {
        let tracker = ProvenanceTracker::new();

        for i in 0..5 {
            let mut record = ProvenanceRecord::new("Linear".to_string(), HashMap::new(), 10, 1);
            record.computation_time = Duration::from_millis((i + 1) * 10);
            tracker.add_record(record);
        }

        let avg = tracker.average_computation_time().unwrap();
        // Average of 10, 20, 30, 40, 50 ms = 30 ms
        assert_eq!(avg.as_millis(), 30);
    }
}
