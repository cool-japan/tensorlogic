//! Debugging utilities for execution tracing and tensor inspection.
//!
//! This module provides comprehensive debugging tools for TensorLogic execution:
//!
//! - **ExecutionTracer**: Record execution flow through computation graphs
//! - **TensorInspector**: Examine intermediate tensor values and statistics
//! - **BreakpointManager**: Pause execution at specific nodes for inspection
//! - **ExecutionRecorder**: Record full execution history for replay and analysis
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_infer::debug::{ExecutionTracer, TensorInspector, BreakpointManager};
//!
//! // Set up tracing
//! let mut tracer = ExecutionTracer::new();
//! tracer.enable();
//!
//! // Add breakpoints
//! let mut breakpoints = BreakpointManager::new();
//! breakpoints.add_node_breakpoint(5);
//!
//! // Execute with debugging
//! // ... execution code ...
//!
//! // Analyze trace
//! let trace = tracer.get_trace();
//! for entry in trace.entries() {
//!     println!("Node {}: {}ms", entry.node_id, entry.duration_ms());
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Execution trace entry recording a single operation.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Unique entry ID
    pub entry_id: usize,
    /// Node ID in the computation graph
    pub node_id: usize,
    /// Operation name
    pub operation: String,
    /// Start timestamp
    pub start_time: Instant,
    /// Duration of execution
    pub duration: Duration,
    /// Input tensor IDs
    pub input_ids: Vec<usize>,
    /// Output tensor IDs
    pub output_ids: Vec<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TraceEntry {
    /// Get duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Get duration in microseconds.
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }
}

/// Execution trace containing recorded operations.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    entries: Vec<TraceEntry>,
    total_duration: Duration,
    graph_id: Option<usize>,
}

impl ExecutionTrace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_duration: Duration::ZERO,
            graph_id: None,
        }
    }

    /// Set the graph ID for this trace.
    pub fn with_graph_id(mut self, graph_id: usize) -> Self {
        self.graph_id = Some(graph_id);
        self
    }

    /// Add a trace entry.
    pub fn add_entry(&mut self, entry: TraceEntry) {
        self.total_duration += entry.duration;
        self.entries.push(entry);
    }

    /// Get all trace entries.
    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    /// Get the total execution duration.
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Get the total duration in milliseconds.
    pub fn total_duration_ms(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1000.0
    }

    /// Get entries for a specific node.
    pub fn entries_for_node(&self, node_id: usize) -> Vec<&TraceEntry> {
        self.entries
            .iter()
            .filter(|e| e.node_id == node_id)
            .collect()
    }

    /// Get the critical path (longest chain of dependent operations).
    pub fn critical_path(&self) -> Vec<&TraceEntry> {
        // Simple implementation: return entries in execution order
        // TODO: Implement proper critical path analysis with dependency tracking
        self.entries.iter().collect()
    }

    /// Get operations sorted by duration (slowest first).
    pub fn slowest_operations(&self, limit: usize) -> Vec<&TraceEntry> {
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
        sorted.into_iter().take(limit).collect()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> TraceSummary {
        TraceSummary::from_trace(self)
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for an execution trace.
#[derive(Debug, Clone)]
pub struct TraceSummary {
    /// Total number of operations
    pub total_operations: usize,
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Average operation time in milliseconds
    pub avg_time_ms: f64,
    /// Slowest operation time in milliseconds
    pub max_time_ms: f64,
    /// Fastest operation time in milliseconds
    pub min_time_ms: f64,
    /// Operation counts by type
    pub operation_counts: HashMap<String, usize>,
}

impl TraceSummary {
    /// Create a summary from a trace.
    pub fn from_trace(trace: &ExecutionTrace) -> Self {
        let entries = trace.entries();
        let total_operations = entries.len();

        let total_time_ms = trace.total_duration_ms();
        let avg_time_ms = if total_operations > 0 {
            total_time_ms / total_operations as f64
        } else {
            0.0
        };

        let max_time_ms = entries.iter().map(|e| e.duration_ms()).fold(0.0, f64::max);
        let min_time_ms = entries
            .iter()
            .map(|e| e.duration_ms())
            .fold(f64::MAX, f64::min);

        let mut operation_counts: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            *operation_counts.entry(entry.operation.clone()).or_insert(0) += 1;
        }

        Self {
            total_operations,
            total_time_ms,
            avg_time_ms,
            max_time_ms,
            min_time_ms,
            operation_counts,
        }
    }
}

impl fmt::Display for TraceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Execution Trace Summary")?;
        writeln!(f, "=======================")?;
        writeln!(f, "Total operations: {}", self.total_operations)?;
        writeln!(f, "Total time: {:.2} ms", self.total_time_ms)?;
        writeln!(f, "Average time: {:.2} ms", self.avg_time_ms)?;
        writeln!(f, "Max time: {:.2} ms", self.max_time_ms)?;
        writeln!(f, "Min time: {:.2} ms", self.min_time_ms)?;
        writeln!(f, "\nOperation Counts:")?;
        let mut sorted_ops: Vec<_> = self.operation_counts.iter().collect();
        sorted_ops.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        for (op, count) in sorted_ops {
            writeln!(f, "  {}: {}", op, count)?;
        }
        Ok(())
    }
}

/// Execution tracer for recording operation flow.
pub struct ExecutionTracer {
    enabled: bool,
    current_trace: ExecutionTrace,
    traces: Vec<ExecutionTrace>,
    next_entry_id: usize,
}

impl ExecutionTracer {
    /// Create a new execution tracer.
    pub fn new() -> Self {
        Self {
            enabled: false,
            current_trace: ExecutionTrace::new(),
            traces: Vec::new(),
            next_entry_id: 0,
        }
    }

    /// Enable tracing.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable tracing.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if tracing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start a new trace (finalizes current trace if any).
    pub fn start_trace(&mut self, graph_id: Option<usize>) {
        if !self.current_trace.entries.is_empty() {
            self.finalize_trace();
        }
        self.current_trace = ExecutionTrace::new();
        if let Some(id) = graph_id {
            self.current_trace.graph_id = Some(id);
        }
    }

    /// Finalize the current trace and store it.
    pub fn finalize_trace(&mut self) {
        if !self.current_trace.entries.is_empty() {
            let trace = std::mem::take(&mut self.current_trace);
            self.traces.push(trace);
        }
    }

    /// Record the start of an operation.
    pub fn record_operation_start(
        &mut self,
        _node_id: usize,
        _operation: impl Into<String>,
        _input_ids: Vec<usize>,
    ) -> OperationHandle {
        if !self.enabled {
            return OperationHandle {
                entry_id: None,
                start_time: Instant::now(),
            };
        }

        let entry_id = self.next_entry_id;
        self.next_entry_id += 1;

        OperationHandle {
            entry_id: Some(entry_id),
            start_time: Instant::now(),
        }
    }

    /// Record the end of an operation.
    pub fn record_operation_end(
        &mut self,
        handle: OperationHandle,
        node_id: usize,
        operation: impl Into<String>,
        input_ids: Vec<usize>,
        output_ids: Vec<usize>,
        metadata: HashMap<String, String>,
    ) {
        if !self.enabled || handle.entry_id.is_none() {
            return;
        }

        let duration = handle.start_time.elapsed();
        let entry = TraceEntry {
            entry_id: handle.entry_id.unwrap(),
            node_id,
            operation: operation.into(),
            start_time: handle.start_time,
            duration,
            input_ids,
            output_ids,
            metadata,
        };

        self.current_trace.add_entry(entry);
    }

    /// Get the current trace.
    pub fn get_trace(&self) -> &ExecutionTrace {
        &self.current_trace
    }

    /// Get all recorded traces.
    pub fn get_all_traces(&self) -> &[ExecutionTrace] {
        &self.traces
    }

    /// Clear all traces.
    pub fn clear(&mut self) {
        self.current_trace = ExecutionTrace::new();
        self.traces.clear();
        self.next_entry_id = 0;
    }
}

impl Default for ExecutionTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for an in-progress operation recording.
pub struct OperationHandle {
    entry_id: Option<usize>,
    start_time: Instant,
}

/// Tensor statistics for inspection.
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Tensor ID
    pub tensor_id: usize,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of elements
    pub num_elements: usize,
    /// Data type
    pub dtype: String,
    /// Minimum value (if computed)
    pub min_value: Option<f64>,
    /// Maximum value (if computed)
    pub max_value: Option<f64>,
    /// Mean value (if computed)
    pub mean_value: Option<f64>,
    /// Standard deviation (if computed)
    pub std_dev: Option<f64>,
    /// Number of NaN values
    pub num_nans: Option<usize>,
    /// Number of infinite values
    pub num_infs: Option<usize>,
}

impl TensorStats {
    /// Create tensor stats with basic information.
    pub fn new(tensor_id: usize, shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        let num_elements = shape.iter().product();
        Self {
            tensor_id,
            shape,
            num_elements,
            dtype: dtype.into(),
            min_value: None,
            max_value: None,
            mean_value: None,
            std_dev: None,
            num_nans: None,
            num_infs: None,
        }
    }

    /// Add computed statistics.
    pub fn with_statistics(
        mut self,
        min: f64,
        max: f64,
        mean: f64,
        std_dev: f64,
        num_nans: usize,
        num_infs: usize,
    ) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self.mean_value = Some(mean);
        self.std_dev = Some(std_dev);
        self.num_nans = Some(num_nans);
        self.num_infs = Some(num_infs);
        self
    }

    /// Check if the tensor has numerical issues.
    pub fn has_numerical_issues(&self) -> bool {
        self.num_nans.unwrap_or(0) > 0 || self.num_infs.unwrap_or(0) > 0
    }
}

impl fmt::Display for TensorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor {} Stats:", self.tensor_id)?;
        writeln!(f, "  Shape: {:?}", self.shape)?;
        writeln!(f, "  Elements: {}", self.num_elements)?;
        writeln!(f, "  DType: {}", self.dtype)?;
        if let Some(min) = self.min_value {
            writeln!(f, "  Min: {:.6}", min)?;
        }
        if let Some(max) = self.max_value {
            writeln!(f, "  Max: {:.6}", max)?;
        }
        if let Some(mean) = self.mean_value {
            writeln!(f, "  Mean: {:.6}", mean)?;
        }
        if let Some(std) = self.std_dev {
            writeln!(f, "  Std Dev: {:.6}", std)?;
        }
        if let Some(nans) = self.num_nans {
            if nans > 0 {
                writeln!(f, "  ⚠️  NaNs: {}", nans)?;
            }
        }
        if let Some(infs) = self.num_infs {
            if infs > 0 {
                writeln!(f, "  ⚠️  Infs: {}", infs)?;
            }
        }
        Ok(())
    }
}

/// Tensor inspector for examining intermediate values.
pub struct TensorInspector {
    enabled: bool,
    tensor_stats: HashMap<usize, TensorStats>,
    watch_list: Vec<usize>,
}

impl TensorInspector {
    /// Create a new tensor inspector.
    pub fn new() -> Self {
        Self {
            enabled: false,
            tensor_stats: HashMap::new(),
            watch_list: Vec::new(),
        }
    }

    /// Enable inspection.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable inspection.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if inspection is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Add a tensor to the watch list.
    pub fn watch(&mut self, tensor_id: usize) {
        if !self.watch_list.contains(&tensor_id) {
            self.watch_list.push(tensor_id);
        }
    }

    /// Remove a tensor from the watch list.
    pub fn unwatch(&mut self, tensor_id: usize) {
        self.watch_list.retain(|&id| id != tensor_id);
    }

    /// Clear the watch list.
    pub fn clear_watch_list(&mut self) {
        self.watch_list.clear();
    }

    /// Check if a tensor should be inspected.
    pub fn should_inspect(&self, tensor_id: usize) -> bool {
        self.enabled && (self.watch_list.is_empty() || self.watch_list.contains(&tensor_id))
    }

    /// Record tensor statistics.
    pub fn record_stats(&mut self, stats: TensorStats) {
        if !self.enabled {
            return;
        }
        self.tensor_stats.insert(stats.tensor_id, stats);
    }

    /// Get statistics for a tensor.
    pub fn get_stats(&self, tensor_id: usize) -> Option<&TensorStats> {
        self.tensor_stats.get(&tensor_id)
    }

    /// Get all recorded statistics.
    pub fn get_all_stats(&self) -> &HashMap<usize, TensorStats> {
        &self.tensor_stats
    }

    /// Find tensors with numerical issues.
    pub fn find_problematic_tensors(&self) -> Vec<&TensorStats> {
        self.tensor_stats
            .values()
            .filter(|stats| stats.has_numerical_issues())
            .collect()
    }

    /// Clear all recorded statistics.
    pub fn clear(&mut self) {
        self.tensor_stats.clear();
    }
}

impl Default for TensorInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Breakpoint type for execution control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Breakpoint {
    /// Break at a specific node
    Node(usize),
    /// Break at a specific operation type
    Operation(String),
    /// Break when a tensor has numerical issues
    NumericalIssue,
    /// Break when execution time exceeds threshold (in microseconds)
    TimeThreshold(u64),
    /// Conditional breakpoint with custom predicate
    Conditional(String), // Store predicate as string for simplicity
}

/// Breakpoint hit information.
#[derive(Debug, Clone)]
pub struct BreakpointHit {
    /// The breakpoint that was hit
    pub breakpoint: Breakpoint,
    /// Node ID where execution paused
    pub node_id: usize,
    /// Current execution time in microseconds
    pub elapsed_us: u64,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Manager for execution breakpoints.
pub struct BreakpointManager {
    enabled: bool,
    breakpoints: Vec<Breakpoint>,
    hits: Vec<BreakpointHit>,
    continue_execution: bool,
}

impl BreakpointManager {
    /// Create a new breakpoint manager.
    pub fn new() -> Self {
        Self {
            enabled: false,
            breakpoints: Vec::new(),
            hits: Vec::new(),
            continue_execution: true,
        }
    }

    /// Enable breakpoint checking.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable breakpoint checking.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if breakpoint checking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Add a node breakpoint.
    pub fn add_node_breakpoint(&mut self, node_id: usize) {
        self.breakpoints.push(Breakpoint::Node(node_id));
    }

    /// Add an operation breakpoint.
    pub fn add_operation_breakpoint(&mut self, operation: impl Into<String>) {
        self.breakpoints
            .push(Breakpoint::Operation(operation.into()));
    }

    /// Add a numerical issue breakpoint.
    pub fn add_numerical_issue_breakpoint(&mut self) {
        self.breakpoints.push(Breakpoint::NumericalIssue);
    }

    /// Add a time threshold breakpoint.
    pub fn add_time_threshold_breakpoint(&mut self, threshold_us: u64) {
        self.breakpoints
            .push(Breakpoint::TimeThreshold(threshold_us));
    }

    /// Remove a breakpoint.
    pub fn remove_breakpoint(&mut self, breakpoint: &Breakpoint) {
        self.breakpoints.retain(|bp| bp != breakpoint);
    }

    /// Clear all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Get all breakpoints.
    pub fn get_breakpoints(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Check if execution should break at this point.
    pub fn should_break(
        &mut self,
        node_id: usize,
        operation: &str,
        elapsed_us: u64,
        has_numerical_issue: bool,
    ) -> Option<BreakpointHit> {
        if !self.enabled || !self.continue_execution {
            return None;
        }

        for breakpoint in &self.breakpoints {
            let should_break = match breakpoint {
                Breakpoint::Node(bp_node_id) => *bp_node_id == node_id,
                Breakpoint::Operation(bp_op) => bp_op == operation,
                Breakpoint::NumericalIssue => has_numerical_issue,
                Breakpoint::TimeThreshold(threshold) => elapsed_us > *threshold,
                Breakpoint::Conditional(_) => false, // Not implemented yet
            };

            if should_break {
                let hit = BreakpointHit {
                    breakpoint: breakpoint.clone(),
                    node_id,
                    elapsed_us,
                    context: HashMap::new(),
                };
                self.hits.push(hit.clone());
                self.continue_execution = false;
                return Some(hit);
            }
        }

        None
    }

    /// Continue execution after a breakpoint hit.
    pub fn continue_execution(&mut self) {
        self.continue_execution = true;
    }

    /// Get all breakpoint hits.
    pub fn get_hits(&self) -> &[BreakpointHit] {
        &self.hits
    }

    /// Clear all breakpoint hits.
    pub fn clear_hits(&mut self) {
        self.hits.clear();
    }
}

impl Default for BreakpointManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Full execution recorder for replay and analysis.
pub struct ExecutionRecorder {
    enabled: bool,
    tracer: ExecutionTracer,
    inspector: TensorInspector,
    breakpoints: BreakpointManager,
}

impl ExecutionRecorder {
    /// Create a new execution recorder.
    pub fn new() -> Self {
        Self {
            enabled: false,
            tracer: ExecutionTracer::new(),
            inspector: TensorInspector::new(),
            breakpoints: BreakpointManager::new(),
        }
    }

    /// Enable all recording features.
    pub fn enable(&mut self) {
        self.enabled = true;
        self.tracer.enable();
        self.inspector.enable();
        self.breakpoints.enable();
    }

    /// Disable all recording features.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.tracer.disable();
        self.inspector.disable();
        self.breakpoints.disable();
    }

    /// Get the tracer.
    pub fn tracer(&mut self) -> &mut ExecutionTracer {
        &mut self.tracer
    }

    /// Get the inspector.
    pub fn inspector(&mut self) -> &mut TensorInspector {
        &mut self.inspector
    }

    /// Get the breakpoint manager.
    pub fn breakpoints(&mut self) -> &mut BreakpointManager {
        &mut self.breakpoints
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.tracer.clear();
        self.inspector.clear();
        self.breakpoints.clear_hits();
    }

    /// Generate a comprehensive execution report.
    pub fn generate_report(&self) -> ExecutionReport {
        ExecutionReport {
            trace_summary: self.tracer.get_trace().summary(),
            problematic_tensors: self.inspector.find_problematic_tensors().len(),
            breakpoint_hits: self.breakpoints.get_hits().len(),
        }
    }
}

impl Default for ExecutionRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive execution report.
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    /// Trace summary
    pub trace_summary: TraceSummary,
    /// Number of tensors with numerical issues
    pub problematic_tensors: usize,
    /// Number of breakpoint hits
    pub breakpoint_hits: usize,
}

impl fmt::Display for ExecutionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.trace_summary)?;
        writeln!(f, "\nDebug Information:")?;
        writeln!(f, "  Problematic tensors: {}", self.problematic_tensors)?;
        writeln!(f, "  Breakpoint hits: {}", self.breakpoint_hits)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_tracer() {
        let mut tracer = ExecutionTracer::new();
        assert!(!tracer.is_enabled());

        tracer.enable();
        assert!(tracer.is_enabled());

        tracer.start_trace(Some(1));
        let handle = tracer.record_operation_start(0, "einsum", vec![0, 1]);
        std::thread::sleep(Duration::from_millis(10));
        tracer.record_operation_end(handle, 0, "einsum", vec![0, 1], vec![2], HashMap::new());

        let trace = tracer.get_trace();
        assert_eq!(trace.entries().len(), 1);
        assert!(trace.total_duration_ms() >= 10.0);
    }

    #[test]
    fn test_trace_summary() {
        let mut trace = ExecutionTrace::new();
        let entry = TraceEntry {
            entry_id: 0,
            node_id: 0,
            operation: "einsum".to_string(),
            start_time: Instant::now(),
            duration: Duration::from_millis(10),
            input_ids: vec![0],
            output_ids: vec![1],
            metadata: HashMap::new(),
        };
        trace.add_entry(entry);

        let summary = trace.summary();
        assert_eq!(summary.total_operations, 1);
        assert!(summary.total_time_ms >= 10.0);
    }

    #[test]
    fn test_tensor_inspector() {
        let mut inspector = TensorInspector::new();
        inspector.enable();

        let stats =
            TensorStats::new(0, vec![2, 3], "f64").with_statistics(0.0, 1.0, 0.5, 0.25, 0, 0);

        inspector.record_stats(stats.clone());
        assert_eq!(inspector.get_stats(0).unwrap().tensor_id, 0);
        assert!(!stats.has_numerical_issues());
    }

    #[test]
    fn test_tensor_numerical_issues() {
        let stats = TensorStats::new(0, vec![2, 3], "f64").with_statistics(
            0.0,
            f64::INFINITY,
            0.5,
            0.25,
            1,
            1,
        );

        assert!(stats.has_numerical_issues());
    }

    #[test]
    fn test_breakpoint_manager() {
        let mut manager = BreakpointManager::new();
        manager.enable();
        manager.add_node_breakpoint(5);

        let hit = manager.should_break(5, "einsum", 1000, false);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().node_id, 5);

        manager.continue_execution();
        let hit2 = manager.should_break(5, "einsum", 1000, false);
        assert!(hit2.is_some());
    }

    #[test]
    fn test_operation_breakpoint() {
        let mut manager = BreakpointManager::new();
        manager.enable();
        manager.add_operation_breakpoint("matmul");

        let hit = manager.should_break(1, "matmul", 1000, false);
        assert!(hit.is_some());

        let no_hit = manager.should_break(2, "add", 1000, false);
        assert!(no_hit.is_none());
    }

    #[test]
    fn test_time_threshold_breakpoint() {
        let mut manager = BreakpointManager::new();
        manager.enable();
        manager.add_time_threshold_breakpoint(5000);

        let no_hit = manager.should_break(1, "op", 4000, false);
        assert!(no_hit.is_none());

        let hit = manager.should_break(1, "op", 6000, false);
        assert!(hit.is_some());
    }

    #[test]
    fn test_numerical_issue_breakpoint() {
        let mut manager = BreakpointManager::new();
        manager.enable();
        manager.add_numerical_issue_breakpoint();

        let no_hit = manager.should_break(1, "op", 1000, false);
        assert!(no_hit.is_none());

        let hit = manager.should_break(1, "op", 1000, true);
        assert!(hit.is_some());
    }

    #[test]
    fn test_execution_recorder() {
        let mut recorder = ExecutionRecorder::new();
        recorder.enable();

        assert!(recorder.tracer().is_enabled());
        assert!(recorder.inspector().is_enabled());
        assert!(recorder.breakpoints().is_enabled());

        recorder.clear();
        let report = recorder.generate_report();
        assert_eq!(report.trace_summary.total_operations, 0);
    }

    #[test]
    fn test_slowest_operations() {
        let mut trace = ExecutionTrace::new();
        for i in 0..5 {
            let entry = TraceEntry {
                entry_id: i,
                node_id: i,
                operation: format!("op{}", i),
                start_time: Instant::now(),
                duration: Duration::from_millis((i as u64 + 1) * 10),
                input_ids: vec![],
                output_ids: vec![],
                metadata: HashMap::new(),
            };
            trace.add_entry(entry);
        }

        let slowest = trace.slowest_operations(3);
        assert_eq!(slowest.len(), 3);
        assert_eq!(slowest[0].node_id, 4); // Slowest
        assert_eq!(slowest[1].node_id, 3);
        assert_eq!(slowest[2].node_id, 2);
    }

    #[test]
    fn test_watch_list() {
        let mut inspector = TensorInspector::new();
        inspector.enable();

        inspector.watch(1);
        inspector.watch(2);

        assert!(inspector.should_inspect(1));
        assert!(inspector.should_inspect(2));
        assert!(!inspector.should_inspect(3));

        inspector.unwatch(1);
        assert!(!inspector.should_inspect(1));
        assert!(inspector.should_inspect(2));

        inspector.clear_watch_list();
        // When watch list is empty, all tensors should be inspected
        assert!(inspector.should_inspect(5));
    }

    #[test]
    fn test_trace_entries_for_node() {
        let mut trace = ExecutionTrace::new();
        for i in 0..3 {
            let entry = TraceEntry {
                entry_id: i,
                node_id: i % 2,
                operation: "op".to_string(),
                start_time: Instant::now(),
                duration: Duration::from_millis(10),
                input_ids: vec![],
                output_ids: vec![],
                metadata: HashMap::new(),
            };
            trace.add_entry(entry);
        }

        let node_0_entries = trace.entries_for_node(0);
        assert_eq!(node_0_entries.len(), 2);

        let node_1_entries = trace.entries_for_node(1);
        assert_eq!(node_1_entries.len(), 1);
    }
}
