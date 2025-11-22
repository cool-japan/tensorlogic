//! Performance monitoring and memory tracking for TensorLogic Python bindings
//!
//! This module provides utilities for measuring memory usage, execution time,
//! and other performance metrics.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Global memory tracker for monitoring allocations
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);

/// Memory usage snapshot
#[pyclass(name = "MemorySnapshot")]
#[derive(Clone)]
pub struct PyMemorySnapshot {
    /// Current allocated bytes
    #[pyo3(get)]
    pub current_bytes: usize,
    /// Peak allocated bytes
    #[pyo3(get)]
    pub peak_bytes: usize,
    /// Total allocation count
    #[pyo3(get)]
    pub allocation_count: u64,
    /// Timestamp when snapshot was taken (for future use)
    #[allow(dead_code)]
    timestamp: Instant,
}

#[pymethods]
impl PyMemorySnapshot {
    /// Get current allocation in megabytes
    #[getter]
    fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak allocation in megabytes
    #[getter]
    fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemorySnapshot(current={:.2} MB, peak={:.2} MB, allocations={})",
            self.current_mb(),
            self.peak_mb(),
            self.allocation_count
        )
    }
}

/// Performance profiler for tracking execution metrics
#[pyclass(name = "Profiler")]
pub struct PyProfiler {
    /// Execution times for named operations
    timings: HashMap<String, Vec<f64>>,
    /// Memory snapshots
    memory_snapshots: Vec<(String, PyMemorySnapshot)>,
    /// Start time of profiling session
    start_time: Instant,
    /// Whether profiling is active
    #[pyo3(get)]
    is_active: bool,
}

#[pymethods]
impl PyProfiler {
    #[new]
    fn new() -> Self {
        PyProfiler {
            timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            start_time: Instant::now(),
            is_active: false,
        }
    }

    /// Start profiling session
    fn start(&mut self) {
        self.is_active = true;
        self.start_time = Instant::now();
        self.timings.clear();
        self.memory_snapshots.clear();
        reset_memory_stats();
    }

    /// Stop profiling session
    fn stop(&mut self) {
        self.is_active = false;
    }

    /// Record an execution time for a named operation
    fn record_time(&mut self, name: String, time_ms: f64) {
        self.timings.entry(name).or_default().push(time_ms);
    }

    /// Take a memory snapshot with a label
    fn snapshot(&mut self, label: String) {
        let snapshot = PyMemorySnapshot {
            current_bytes: TOTAL_ALLOCATED.load(Ordering::Relaxed),
            peak_bytes: PEAK_ALLOCATED.load(Ordering::Relaxed),
            allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
            timestamp: Instant::now(),
        };
        self.memory_snapshots.push((label, snapshot));
    }

    /// Get timing statistics for an operation
    fn get_timing_stats<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        if let Some(times) = self.timings.get(name) {
            if !times.is_empty() {
                let sum: f64 = times.iter().sum();
                let count = times.len() as f64;
                let mean = sum / count;
                let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Calculate standard deviation
                let variance: f64 = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / count;
                let std = variance.sqrt();

                // Calculate percentiles
                let mut sorted = times.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = sorted[sorted.len() / 2];
                let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
                let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

                dict.set_item("count", times.len())?;
                dict.set_item("mean", mean)?;
                dict.set_item("std", std)?;
                dict.set_item("min", min)?;
                dict.set_item("max", max)?;
                dict.set_item("p50", p50)?;
                dict.set_item("p95", p95)?;
                dict.set_item("p99", p99)?;
            }
        }

        Ok(dict)
    }

    /// Get all timing operation names
    fn get_operation_names(&self) -> Vec<String> {
        self.timings.keys().cloned().collect()
    }

    /// Get memory snapshots
    fn get_memory_snapshots(&self) -> Vec<(String, PyMemorySnapshot)> {
        self.memory_snapshots.clone()
    }

    /// Get total elapsed time since profiling started
    fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Generate a summary report
    fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("Performance Profile Summary\n");
        report.push_str(&"=".repeat(50));
        report.push('\n');

        // Timing summary
        if !self.timings.is_empty() {
            report.push_str("\nExecution Times:\n");
            for (name, times) in &self.timings {
                if !times.is_empty() {
                    let sum: f64 = times.iter().sum();
                    let mean = sum / times.len() as f64;
                    report.push_str(&format!(
                        "  {}: {:.4} ms (n={}, total={:.2} ms)\n",
                        name,
                        mean,
                        times.len(),
                        sum
                    ));
                }
            }
        }

        // Memory summary
        if !self.memory_snapshots.is_empty() {
            report.push_str("\nMemory Snapshots:\n");
            for (label, snapshot) in &self.memory_snapshots {
                report.push_str(&format!("  {}: {:.2} MB\n", label, snapshot.current_mb()));
            }
        }

        report.push_str(&format!("\nTotal elapsed: {:.2} ms\n", self.elapsed_ms()));
        report
    }

    fn __repr__(&self) -> String {
        format!(
            "Profiler(operations={}, snapshots={}, active={})",
            self.timings.len(),
            self.memory_snapshots.len(),
            self.is_active
        )
    }
}

/// Timer context for measuring execution time
#[pyclass(name = "Timer")]
pub struct PyTimer {
    name: String,
    start: Option<Instant>,
    elapsed_ms: Option<f64>,
    /// Profiler reference for future use (auto-recording to profiler)
    #[allow(dead_code)]
    profiler: Option<Arc<std::sync::Mutex<PyProfiler>>>,
}

#[pymethods]
impl PyTimer {
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<String>) -> Self {
        PyTimer {
            name: name.unwrap_or_else(|| "timer".to_string()),
            start: None,
            elapsed_ms: None,
            profiler: None,
        }
    }

    /// Start the timer
    fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop the timer and return elapsed time in milliseconds
    fn stop(&mut self) -> f64 {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            self.elapsed_ms = Some(elapsed);
            elapsed
        } else {
            0.0
        }
    }

    /// Get elapsed time in milliseconds
    fn elapsed(&self) -> f64 {
        if let Some(elapsed) = self.elapsed_ms {
            elapsed
        } else if let Some(start) = self.start {
            start.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        }
    }

    /// Context manager enter
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.start = Some(Instant::now());
        slf
    }

    /// Context manager exit
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, pyo3::types::PyAny>,
        _exc_val: &Bound<'_, pyo3::types::PyAny>,
        _exc_tb: &Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        self.stop();
        false
    }

    fn __repr__(&self) -> String {
        if let Some(elapsed) = self.elapsed_ms {
            format!("Timer({}: {:.4} ms)", self.name, elapsed)
        } else if self.start.is_some() {
            format!("Timer({}: running)", self.name)
        } else {
            format!("Timer({}: not started)", self.name)
        }
    }
}

/// Reset memory statistics
fn reset_memory_stats() {
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    PEAK_ALLOCATED.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
}

/// Track memory allocation (called internally)
/// Reserved for future deep integration with allocators
#[allow(dead_code)]
pub fn track_allocation(bytes: usize) {
    let current = TOTAL_ALLOCATED.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_ALLOCATED.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_ALLOCATED.compare_exchange_weak(
            peak,
            current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(p) => peak = p,
        }
    }
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Track memory deallocation (called internally)
/// Reserved for future deep integration with allocators
#[allow(dead_code)]
pub fn track_deallocation(bytes: usize) {
    TOTAL_ALLOCATED.fetch_sub(bytes, Ordering::Relaxed);
}

/// Get current memory snapshot
#[pyfunction(name = "memory_snapshot")]
pub fn py_memory_snapshot() -> PyMemorySnapshot {
    PyMemorySnapshot {
        current_bytes: TOTAL_ALLOCATED.load(Ordering::Relaxed),
        peak_bytes: PEAK_ALLOCATED.load(Ordering::Relaxed),
        allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
        timestamp: Instant::now(),
    }
}

/// Create a new profiler
#[pyfunction(name = "profiler")]
pub fn py_profiler() -> PyProfiler {
    PyProfiler::new()
}

/// Create a new timer
#[pyfunction(name = "timer")]
#[pyo3(signature = (name=None))]
pub fn py_timer(name: Option<String>) -> PyTimer {
    PyTimer::new(name)
}

/// Get system memory information
#[pyfunction(name = "get_memory_info")]
pub fn py_get_memory_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // Get tracked memory
    dict.set_item("tracked_bytes", TOTAL_ALLOCATED.load(Ordering::Relaxed))?;
    dict.set_item("peak_bytes", PEAK_ALLOCATED.load(Ordering::Relaxed))?;
    dict.set_item("allocation_count", ALLOCATION_COUNT.load(Ordering::Relaxed))?;

    // Calculate MB
    let current_mb = TOTAL_ALLOCATED.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let peak_mb = PEAK_ALLOCATED.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    dict.set_item("tracked_mb", current_mb)?;
    dict.set_item("peak_mb", peak_mb)?;

    Ok(dict)
}

/// Reset all memory tracking statistics
#[pyfunction(name = "reset_memory_tracking")]
pub fn py_reset_memory_tracking() {
    reset_memory_stats();
}

/// Register performance module functions
pub fn register_performance_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemorySnapshot>()?;
    m.add_class::<PyProfiler>()?;
    m.add_class::<PyTimer>()?;
    m.add_function(wrap_pyfunction!(py_memory_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(py_profiler, m)?)?;
    m.add_function(wrap_pyfunction!(py_timer, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_memory_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_reset_memory_tracking, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracking() {
        reset_memory_stats();
        track_allocation(1024);
        assert_eq!(TOTAL_ALLOCATED.load(Ordering::Relaxed), 1024);
        track_deallocation(512);
        assert_eq!(TOTAL_ALLOCATED.load(Ordering::Relaxed), 512);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = PyProfiler::new();
        profiler.start();
        profiler.record_time("test".to_string(), 10.0);
        profiler.record_time("test".to_string(), 20.0);
        assert!(profiler.is_active);
        profiler.stop();
        assert!(!profiler.is_active);
    }
}
