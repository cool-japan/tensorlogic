//! Enhanced error diagnostics with helpful suggestions.
//!
//! This module provides rich error messages with context, suggestions,
//! and actionable advice for common mistakes.

use std::fmt;
use tensorlogic_ir::EinsumGraph;

use crate::shape::TensorShape;

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning (non-fatal)
    Warning,
    /// Error (fatal, prevents execution)
    Error,
    /// Critical error (system-level issue)
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl SourceLocation {
    pub fn new() -> Self {
        SourceLocation {
            file: None,
            line: None,
            column: None,
        }
    }

    pub fn with_file(mut self, file: String) -> Self {
        self.file = Some(file);
        self
    }

    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }
}

impl Default for SourceLocation {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}", file)?;
            if let Some(line) = self.line {
                write!(f, ":{}", line)?;
                if let Some(column) = self.column {
                    write!(f, ":{}", column)?;
                }
            }
        } else {
            write!(f, "<unknown>")?;
        }
        Ok(())
    }
}

/// Detailed diagnostic message
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub severity: Severity,
    /// Primary error message
    pub message: String,
    /// Source location
    pub location: Option<SourceLocation>,
    /// Additional context
    pub context: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// Related nodes or operations
    pub related: Vec<String>,
    /// Error code (for documentation lookup)
    pub code: Option<String>,
}

impl Diagnostic {
    /// Create a new diagnostic
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            message: message.into(),
            location: None,
            context: Vec::new(),
            suggestions: Vec::new(),
            related: Vec::new(),
            code: None,
        }
    }

    /// Create an error diagnostic
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// Create a warning diagnostic
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// Create an info diagnostic
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(Severity::Info, message)
    }

    /// Add source location
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add context information
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// Add suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add related information
    pub fn with_related(mut self, related: impl Into<String>) -> Self {
        self.related.push(related.into());
        self
    }

    /// Add error code
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Format as user-friendly string
    pub fn format(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("[{}] {}\n", self.severity, self.message));

        // Location
        if let Some(ref loc) = self.location {
            output.push_str(&format!("  at {}\n", loc));
        }

        // Error code
        if let Some(ref code) = self.code {
            output.push_str(&format!("  code: {}\n", code));
        }

        // Context
        if !self.context.is_empty() {
            output.push_str("\nContext:\n");
            for ctx in &self.context {
                output.push_str(&format!("  {}\n", ctx));
            }
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, suggestion));
            }
        }

        // Related
        if !self.related.is_empty() {
            output.push_str("\nRelated:\n");
            for rel in &self.related {
                output.push_str(&format!("  - {}\n", rel));
            }
        }

        output
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Shape mismatch diagnostic builder
pub struct ShapeMismatchDiagnostic;

impl ShapeMismatchDiagnostic {
    pub fn create(expected: &TensorShape, actual: &TensorShape, operation: &str) -> Diagnostic {
        let mut diag = Diagnostic::error(format!("Shape mismatch in {} operation", operation))
            .with_code("E001")
            .with_context(format!(
                "Expected shape: {:?}, but got: {:?}",
                expected.dims, actual.dims
            ));

        // Add specific suggestions based on the mismatch
        if expected.rank() != actual.rank() {
            diag = diag
                .with_suggestion(format!(
                    "Expected rank {} but got rank {}. Consider reshaping your tensor.",
                    expected.rank(),
                    actual.rank()
                ))
                .with_suggestion(format!(
                    "Use tensor.reshape({:?}) to match the expected shape",
                    expected.dims
                ));
        } else {
            // Same rank, dimension mismatch
            let mismatches: Vec<_> = expected
                .dims
                .iter()
                .zip(actual.dims.iter())
                .enumerate()
                .filter(|(_, (e, a))| e != a)
                .collect();

            for (dim, (exp, act)) in mismatches {
                diag = diag.with_context(format!(
                    "Dimension {} mismatch: expected {:?}, got {:?}",
                    dim, exp, act
                ));
            }

            diag = diag.with_suggestion(
                "Check your input tensor shapes match the expected dimensions".to_string(),
            );
        }

        diag
    }
}

/// Type mismatch diagnostic builder
pub struct TypeMismatchDiagnostic;

impl TypeMismatchDiagnostic {
    pub fn create(expected: &str, actual: &str, context: &str) -> Diagnostic {
        Diagnostic::error(format!("Type mismatch in {}", context))
            .with_code("E002")
            .with_context(format!("Expected type: {}, but got: {}", expected, actual))
            .with_suggestion(format!("Convert your data to {} type", expected))
            .with_suggestion("Check the input data types match the expected types".to_string())
    }
}

/// Node execution diagnostic builder
pub struct NodeExecutionDiagnostic;

impl NodeExecutionDiagnostic {
    pub fn create(node_id: usize, error: &str, graph: &EinsumGraph) -> Diagnostic {
        let mut diag = Diagnostic::error(format!("Failed to execute node {}", node_id))
            .with_code("E003")
            .with_context(error.to_string());

        // Add node information
        if let Some(node) = graph.nodes.get(node_id) {
            diag = diag.with_context(format!("Node operation: {:?}", node.op));

            // Add input information
            if !node.inputs.is_empty() {
                diag = diag.with_context(format!("Input nodes: {:?}", node.inputs));
            }

            // Add suggestions based on operation type
            diag = diag.with_suggestion(
                "Check that all input tensors are properly initialized".to_string(),
            );
            diag = diag.with_suggestion(
                "Verify input tensor shapes are compatible with this operation".to_string(),
            );
        }

        // Add related nodes
        for input_id in graph
            .nodes
            .get(node_id)
            .map(|n| &n.inputs)
            .unwrap_or(&vec![])
        {
            diag = diag.with_related(format!("Input node: {}", input_id));
        }

        diag
    }
}

/// Memory diagnostic builder
pub struct MemoryDiagnostic;

impl MemoryDiagnostic {
    pub fn out_of_memory(requested_bytes: usize, available_bytes: usize) -> Diagnostic {
        let requested_mb = requested_bytes as f64 / (1024.0 * 1024.0);
        let available_mb = available_bytes as f64 / (1024.0 * 1024.0);

        Diagnostic::error("Out of memory")
            .with_code("E004")
            .with_context(format!(
                "Requested: {:.2} MB, Available: {:.2} MB",
                requested_mb, available_mb
            ))
            .with_suggestion("Reduce batch size to lower memory usage".to_string())
            .with_suggestion("Enable streaming execution for large datasets".to_string())
            .with_suggestion("Consider using a machine with more memory".to_string())
            .with_suggestion("Enable memory pooling to reuse allocations".to_string())
    }

    pub fn memory_leak_warning(leaked_bytes: usize) -> Diagnostic {
        let leaked_mb = leaked_bytes as f64 / (1024.0 * 1024.0);

        Diagnostic::warning(format!(
            "Potential memory leak detected: {:.2} MB",
            leaked_mb
        ))
        .with_code("W001")
        .with_suggestion("Check that all tensors are properly released".to_string())
        .with_suggestion("Enable memory profiling to identify the leak source".to_string())
        .with_suggestion("Use memory pooling to manage allocations".to_string())
    }
}

/// Performance diagnostic builder
pub struct PerformanceDiagnostic;

impl PerformanceDiagnostic {
    pub fn slow_operation(
        operation: &str,
        actual_time_ms: f64,
        expected_time_ms: f64,
    ) -> Diagnostic {
        let slowdown = actual_time_ms / expected_time_ms;

        Diagnostic::warning(format!(
            "Slow {} operation: {:.2}x slower than expected",
            operation, slowdown
        ))
        .with_code("W002")
        .with_context(format!(
            "Actual: {:.2}ms, Expected: {:.2}ms",
            actual_time_ms, expected_time_ms
        ))
        .with_suggestion("Enable graph optimization to improve performance".to_string())
        .with_suggestion("Check if operation fusion is enabled".to_string())
        .with_suggestion("Consider using a more powerful device (GPU)".to_string())
        .with_suggestion("Profile the execution to identify bottlenecks".to_string())
    }

    pub fn high_memory_usage(peak_mb: f64, threshold_mb: f64) -> Diagnostic {
        Diagnostic::warning(format!("High memory usage: {:.2} MB", peak_mb))
            .with_code("W003")
            .with_context(format!("Threshold: {:.2} MB", threshold_mb))
            .with_suggestion("Enable memory optimization".to_string())
            .with_suggestion("Reduce batch size".to_string())
            .with_suggestion("Use streaming execution for large datasets".to_string())
    }
}

/// Diagnostic collector for gathering multiple diagnostics
#[derive(Debug, Default)]
pub struct DiagnosticCollector {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a diagnostic
    pub fn add(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity >= Severity::Error)
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }

    /// Format all diagnostics
    pub fn format_all(&self) -> String {
        let mut output = String::new();
        for diag in &self.diagnostics {
            output.push_str(&diag.format());
            output.push('\n');
        }

        output.push_str(&format!(
            "\nSummary: {} error(s), {} warning(s)\n",
            self.error_count(),
            self.warning_count()
        ));

        output
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let diag = Diagnostic::error("Test error")
            .with_code("E001")
            .with_context("Additional context")
            .with_suggestion("Try this fix");

        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.message, "Test error");
        assert_eq!(diag.code, Some("E001".to_string()));
        assert_eq!(diag.context.len(), 1);
        assert_eq!(diag.suggestions.len(), 1);
    }

    #[test]
    fn test_shape_mismatch_diagnostic() {
        let expected = TensorShape::static_shape(vec![64, 128]);
        let actual = TensorShape::static_shape(vec![64, 256]);

        let diag = ShapeMismatchDiagnostic::create(&expected, &actual, "matmul");

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("Shape mismatch"));
        assert!(!diag.suggestions.is_empty());
    }

    #[test]
    fn test_type_mismatch_diagnostic() {
        let diag = TypeMismatchDiagnostic::create("f32", "f64", "tensor operation");

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("Type mismatch"));
        assert_eq!(diag.code, Some("E002".to_string()));
    }

    #[test]
    fn test_memory_diagnostic() {
        let diag = MemoryDiagnostic::out_of_memory(1024 * 1024 * 1024, 512 * 1024 * 1024);

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("Out of memory"));
        assert!(!diag.suggestions.is_empty());
    }

    #[test]
    fn test_performance_diagnostic() {
        let diag = PerformanceDiagnostic::slow_operation("einsum", 100.0, 50.0);

        assert_eq!(diag.severity, Severity::Warning);
        assert!(diag.message.contains("Slow"));
        assert!(diag.message.contains("2.00x"));
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagnosticCollector::new();

        collector.add(Diagnostic::error("Error 1"));
        collector.add(Diagnostic::warning("Warning 1"));
        collector.add(Diagnostic::error("Error 2"));

        assert_eq!(collector.error_count(), 2);
        assert_eq!(collector.warning_count(), 1);
        assert!(collector.has_errors());

        let formatted = collector.format_all();
        assert!(formatted.contains("2 error(s), 1 warning(s)"));
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new()
            .with_file("test.rs".to_string())
            .with_line(42)
            .with_column(10);

        assert_eq!(loc.to_string(), "test.rs:42:10");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }
}
