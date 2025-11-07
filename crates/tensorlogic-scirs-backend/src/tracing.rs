//! Execution tracing and debugging support.
//!
//! This module provides detailed logging and tracing of tensor operations,
//! enabling debugging of execution issues and performance analysis.

use scirs2_core::ndarray::ArrayD;
use std::fmt;
use std::time::{Duration, Instant};
use tensorlogic_ir::{EinsumNode, OpType};

/// Trace level for execution logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraceLevel {
    /// No tracing
    None = 0,
    /// Trace errors only
    Error = 1,
    /// Trace warnings and errors
    Warn = 2,
    /// Trace key operations
    Info = 3,
    /// Trace all operations
    Debug = 4,
    /// Trace everything including tensor values
    Trace = 5,
}

/// A single trace event during execution
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Sequential event ID
    pub id: usize,
    /// When the event occurred (relative to trace start)
    pub timestamp: Duration,
    /// Operation being traced
    pub operation: String,
    /// Node index in the graph (if applicable)
    pub node_index: Option<usize>,
    /// Input tensor IDs
    pub inputs: Vec<String>,
    /// Output tensor IDs
    pub outputs: Vec<String>,
    /// Duration of the operation
    pub duration: Option<Duration>,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Additional metadata
    pub metadata: Option<String>,
    /// Event level
    pub level: TraceLevel,
}

impl fmt::Display for TraceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:6}] {:?} {} ", self.id, self.level, self.operation)?;
        if let Some(node_idx) = self.node_index {
            write!(f, "(node {}) ", node_idx)?;
        }
        write!(f, "{:?} -> {:?}", self.inputs, self.outputs)?;
        if let Some(dur) = self.duration {
            write!(f, " [{:?}]", dur)?;
        }
        Ok(())
    }
}

/// Execution tracer for debugging and profiling
pub struct ExecutionTracer {
    /// Trace level
    level: TraceLevel,
    /// Collected trace events
    events: Vec<TraceEvent>,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Next event ID
    next_id: usize,
    /// Whether tracing is enabled
    enabled: bool,
}

impl ExecutionTracer {
    /// Create a new tracer with the specified level
    pub fn new(level: TraceLevel) -> Self {
        Self {
            level,
            events: Vec::new(),
            start_time: Instant::now(),
            next_id: 0,
            enabled: level != TraceLevel::None,
        }
    }

    /// Create a disabled tracer
    pub fn disabled() -> Self {
        Self::new(TraceLevel::None)
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current trace level
    pub fn level(&self) -> TraceLevel {
        self.level
    }

    /// Set the trace level
    pub fn set_level(&mut self, level: TraceLevel) {
        self.level = level;
        self.enabled = level != TraceLevel::None;
    }

    /// Check if a specific level should be traced
    pub fn should_trace(&self, level: TraceLevel) -> bool {
        self.enabled && level <= self.level
    }

    /// Start tracing an operation
    pub fn start_operation(
        &mut self,
        operation: impl Into<String>,
        node_index: Option<usize>,
        inputs: Vec<String>,
        level: TraceLevel,
    ) -> Option<TraceHandle> {
        if !self.should_trace(level) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        Some(TraceHandle {
            id,
            operation: operation.into(),
            node_index,
            inputs,
            start_time: Instant::now(),
            level,
        })
    }

    /// Finish tracing an operation
    pub fn finish_operation(
        &mut self,
        handle: TraceHandle,
        outputs: Vec<String>,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        metadata: Option<String>,
    ) {
        if !self.should_trace(handle.level) {
            return;
        }

        let duration = handle.start_time.elapsed();
        let timestamp = self.start_time.elapsed();

        let event = TraceEvent {
            id: handle.id,
            timestamp,
            operation: handle.operation,
            node_index: handle.node_index,
            inputs: handle.inputs,
            outputs,
            duration: Some(duration),
            input_shapes,
            output_shapes,
            metadata,
            level: handle.level,
        };

        self.events.push(event);
    }

    /// Record a simple event without timing
    pub fn record_event(
        &mut self,
        operation: impl Into<String>,
        level: TraceLevel,
        metadata: Option<String>,
    ) {
        if !self.should_trace(level) {
            return;
        }

        let id = self.next_id;
        self.next_id += 1;

        let event = TraceEvent {
            id,
            timestamp: self.start_time.elapsed(),
            operation: operation.into(),
            node_index: None,
            inputs: vec![],
            outputs: vec![],
            duration: None,
            input_shapes: vec![],
            output_shapes: vec![],
            metadata,
            level,
        };

        self.events.push(event);
    }

    /// Get all trace events
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Clear all trace events
    pub fn clear(&mut self) {
        self.events.clear();
        self.next_id = 0;
        self.start_time = Instant::now();
    }

    /// Get statistics about traced operations
    pub fn stats(&self) -> TraceStats {
        let mut total_ops = 0;
        let mut total_duration = Duration::ZERO;
        let mut op_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for event in &self.events {
            total_ops += 1;
            if let Some(dur) = event.duration {
                total_duration += dur;
            }
            *op_counts.entry(event.operation.clone()).or_insert(0) += 1;
        }

        TraceStats {
            total_operations: total_ops,
            total_duration,
            operation_counts: op_counts,
        }
    }

    /// Format trace as a string
    pub fn format_trace(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Execution Trace ===\n");
        for event in &self.events {
            output.push_str(&format!("{}\n", event));
        }
        output.push_str("\n=== Statistics ===\n");
        let stats = self.stats();
        output.push_str(&format!("{}", stats));
        output
    }
}

impl Default for ExecutionTracer {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Handle for an in-progress trace operation
pub struct TraceHandle {
    id: usize,
    operation: String,
    node_index: Option<usize>,
    inputs: Vec<String>,
    start_time: Instant,
    level: TraceLevel,
}

/// Statistics collected from trace events
#[derive(Debug, Clone)]
pub struct TraceStats {
    /// Total number of operations traced
    pub total_operations: usize,
    /// Total execution time
    pub total_duration: Duration,
    /// Count of each operation type
    pub operation_counts: std::collections::HashMap<String, usize>,
}

impl fmt::Display for TraceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Total operations: {}", self.total_operations)?;
        writeln!(f, "Total duration: {:?}", self.total_duration)?;
        writeln!(f, "\nOperation counts:")?;
        let mut counts: Vec<_> = self.operation_counts.iter().collect();
        counts.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        for (op, count) in counts {
            writeln!(f, "  {}: {}", op, count)?;
        }
        Ok(())
    }
}

/// Helper to extract operation name from EinsumNode
pub fn operation_name(node: &EinsumNode) -> String {
    match &node.op {
        OpType::Einsum { spec } => format!("einsum({})", spec),
        OpType::ElemUnary { op } => format!("unary({:?})", op),
        OpType::ElemBinary { op, .. } => format!("binary({:?})", op),
        OpType::Reduce { op, axes } => format!("reduce({:?}, axes={:?})", op, axes),
    }
}

/// Helper to get tensor shape as Vec<usize>
pub fn tensor_shape(tensor: &ArrayD<f64>) -> Vec<usize> {
    tensor.shape().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_levels() {
        let mut tracer = ExecutionTracer::new(TraceLevel::Info);
        assert!(tracer.should_trace(TraceLevel::Error));
        assert!(tracer.should_trace(TraceLevel::Warn));
        assert!(tracer.should_trace(TraceLevel::Info));
        assert!(!tracer.should_trace(TraceLevel::Debug));
        assert!(!tracer.should_trace(TraceLevel::Trace));

        tracer.set_level(TraceLevel::Debug);
        assert!(tracer.should_trace(TraceLevel::Debug));
    }

    #[test]
    fn test_trace_operation() {
        let mut tracer = ExecutionTracer::new(TraceLevel::Info);

        let handle = tracer.start_operation(
            "test_op",
            Some(0),
            vec!["input1".to_string()],
            TraceLevel::Info,
        );
        assert!(handle.is_some());

        tracer.finish_operation(
            handle.unwrap(),
            vec!["output1".to_string()],
            vec![vec![2, 3]],
            vec![vec![2, 3]],
            None,
        );

        assert_eq!(tracer.events().len(), 1);
        let event = &tracer.events()[0];
        assert_eq!(event.operation, "test_op");
        assert_eq!(event.node_index, Some(0));
    }

    #[test]
    fn test_tracer_disabled() {
        let mut tracer = ExecutionTracer::disabled();
        assert!(!tracer.is_enabled());

        let handle = tracer.start_operation("test_op", None, vec![], TraceLevel::Info);
        assert!(handle.is_none());
        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_trace_stats() {
        let mut tracer = ExecutionTracer::new(TraceLevel::Info);

        for i in 0..5 {
            let handle = tracer
                .start_operation("einsum", Some(i), vec![], TraceLevel::Info)
                .unwrap();
            tracer.finish_operation(handle, vec![], vec![], vec![], None);
        }

        for i in 0..3 {
            let handle = tracer
                .start_operation("reduce", Some(i + 5), vec![], TraceLevel::Info)
                .unwrap();
            tracer.finish_operation(handle, vec![], vec![], vec![], None);
        }

        let stats = tracer.stats();
        assert_eq!(stats.total_operations, 8);
        assert_eq!(stats.operation_counts.get("einsum"), Some(&5));
        assert_eq!(stats.operation_counts.get("reduce"), Some(&3));
    }

    #[test]
    fn test_trace_clear() {
        let mut tracer = ExecutionTracer::new(TraceLevel::Info);

        tracer.record_event("test", TraceLevel::Info, None);
        assert_eq!(tracer.events().len(), 1);

        tracer.clear();
        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_trace_format() {
        let mut tracer = ExecutionTracer::new(TraceLevel::Info);
        tracer.record_event("test_op", TraceLevel::Info, Some("metadata".to_string()));

        let output = tracer.format_trace();
        assert!(output.contains("Execution Trace"));
        assert!(output.contains("test_op"));
        assert!(output.contains("Statistics"));
    }
}
