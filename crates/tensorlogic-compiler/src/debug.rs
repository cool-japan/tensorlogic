//! Debug utilities for tracing compilation.
//!
//! This module provides tools for debugging and understanding the compilation process,
//! including intermediate state tracking, step-by-step tracing, and detailed logging.

use std::collections::HashMap;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::context::CompilerContext;

/// Compilation trace for debugging.
///
/// Captures the state of compilation at each major step, allowing
/// developers to understand how expressions are transformed into tensor graphs.
#[derive(Debug, Clone)]
pub struct CompilationTrace {
    /// Original expression before compilation
    pub input_expr: String,
    /// Compilation steps with intermediate states
    pub steps: Vec<CompilationStep>,
    /// Final compiled graph (if successful)
    pub final_graph: Option<String>,
    /// Errors encountered during compilation
    pub errors: Vec<String>,
    /// Total compilation time (if measured)
    pub duration_ms: Option<f64>,
}

/// A single step in the compilation process.
#[derive(Debug, Clone)]
pub struct CompilationStep {
    /// Step number (0-indexed)
    pub step_num: usize,
    /// Name of this compilation phase
    pub phase: String,
    /// Description of what happened
    pub description: String,
    /// State snapshot at this step
    pub state: StepState,
    /// Duration of this step (if measured)
    pub duration_us: Option<u64>,
}

/// Snapshot of compiler state at a specific step.
#[derive(Debug, Clone)]
pub struct StepState {
    /// Number of tensors in the graph
    pub tensor_count: usize,
    /// Number of nodes in the graph
    pub node_count: usize,
    /// Number of domains defined
    pub domain_count: usize,
    /// Number of bound variables
    pub bound_vars: usize,
    /// Number of axis assignments
    pub axis_assignments: usize,
    /// Additional custom data
    pub metadata: HashMap<String, String>,
}

impl CompilationTrace {
    /// Create a new empty trace.
    pub fn new(input_expr: &TLExpr) -> Self {
        Self {
            input_expr: format!("{:?}", input_expr),
            steps: Vec::new(),
            final_graph: None,
            errors: Vec::new(),
            duration_ms: None,
        }
    }

    /// Add a compilation step to the trace.
    pub fn add_step(
        &mut self,
        phase: impl Into<String>,
        description: impl Into<String>,
        ctx: &CompilerContext,
        graph: &EinsumGraph,
    ) {
        let state = StepState {
            tensor_count: graph.tensors.len(),
            node_count: graph.nodes.len(),
            domain_count: ctx.domains.len(),
            bound_vars: ctx.var_to_domain.len(),
            axis_assignments: ctx.var_to_axis.len(),
            metadata: HashMap::new(),
        };

        self.steps.push(CompilationStep {
            step_num: self.steps.len(),
            phase: phase.into(),
            description: description.into(),
            state,
            duration_us: None,
        });
    }

    /// Add an error to the trace.
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
    }

    /// Set the final compiled graph.
    pub fn set_final_graph(&mut self, graph: &EinsumGraph) {
        self.final_graph = Some(format!("{:?}", graph));
    }

    /// Set the total compilation duration.
    pub fn set_duration(&mut self, duration_ms: f64) {
        self.duration_ms = Some(duration_ms);
    }

    /// Print a summary of the compilation trace.
    pub fn print_summary(&self) {
        println!("=== Compilation Trace Summary ===");
        println!("Input: {}", truncate(&self.input_expr, 100));
        println!("Steps: {}", self.steps.len());
        println!("Errors: {}", self.errors.len());

        if let Some(dur) = self.duration_ms {
            println!("Duration: {:.3}ms", dur);
        }

        println!("\n--- Steps ---");
        for step in &self.steps {
            println!(
                "{:2}. {} - {} (T:{}, N:{})",
                step.step_num,
                step.phase,
                step.description,
                step.state.tensor_count,
                step.state.node_count
            );
        }

        if !self.errors.is_empty() {
            println!("\n--- Errors ---");
            for (i, error) in self.errors.iter().enumerate() {
                println!("{}. {}", i + 1, error);
            }
        }

        if let Some(ref graph) = self.final_graph {
            println!("\n--- Final Graph ---");
            println!("{}", truncate(graph, 200));
        }

        println!("================================");
    }

    /// Generate a detailed report with all intermediate states.
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔════════════════════════════════════════╗\n");
        report.push_str("║   COMPILATION TRACE - DETAILED REPORT   ║\n");
        report.push_str("╚════════════════════════════════════════╝\n\n");

        report.push_str(&format!("Input Expression:\n  {}\n\n", self.input_expr));

        if let Some(dur) = self.duration_ms {
            report.push_str(&format!("Total Duration: {:.3}ms\n\n", dur));
        }

        report.push_str("Compilation Steps:\n");
        report.push_str("─────────────────────────────────────────\n\n");

        for step in &self.steps {
            report.push_str(&format!("Step {}: {}\n", step.step_num, step.phase));
            report.push_str(&format!("  Description: {}\n", step.description));
            report.push_str("  State:\n");
            report.push_str(&format!("    Tensors: {}\n", step.state.tensor_count));
            report.push_str(&format!("    Nodes: {}\n", step.state.node_count));
            report.push_str(&format!("    Domains: {}\n", step.state.domain_count));
            report.push_str(&format!("    Bound Variables: {}\n", step.state.bound_vars));
            report.push_str(&format!(
                "    Axis Assignments: {}\n",
                step.state.axis_assignments
            ));

            if !step.state.metadata.is_empty() {
                report.push_str("    Metadata:\n");
                for (key, value) in &step.state.metadata {
                    report.push_str(&format!("      {}: {}\n", key, value));
                }
            }

            if let Some(dur) = step.duration_us {
                report.push_str(&format!("  Duration: {}μs\n", dur));
            }

            report.push('\n');
        }

        if !self.errors.is_empty() {
            report.push_str("Errors Encountered:\n");
            report.push_str("─────────────────────────────────────────\n");
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, error));
            }
            report.push('\n');
        }

        if let Some(ref graph) = self.final_graph {
            report.push_str("Final Graph:\n");
            report.push_str("─────────────────────────────────────────\n");
            report.push_str(graph);
            report.push('\n');
        }

        report
    }
}

/// Helper function to truncate long strings.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Compilation tracer that can be enabled/disabled.
///
/// Use this to instrument compilation with tracing:
///
/// ```ignore
/// let mut tracer = CompilationTracer::new(true); // enabled
/// tracer.start(&expr);
///
/// // During compilation:
/// tracer.record_step("Parse", "Parsed expression", &ctx, &graph);
/// tracer.record_step("Optimize", "Applied CSE", &ctx, &graph);
///
/// let trace = tracer.finish(&graph);
/// trace.print_summary();
/// ```
pub struct CompilationTracer {
    enabled: bool,
    trace: Option<CompilationTrace>,
    start_time: Option<std::time::Instant>,
}

impl CompilationTracer {
    /// Create a new tracer.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            trace: None,
            start_time: None,
        }
    }

    /// Start tracing for the given expression.
    pub fn start(&mut self, expr: &TLExpr) {
        if self.enabled {
            self.trace = Some(CompilationTrace::new(expr));
            self.start_time = Some(std::time::Instant::now());
        }
    }

    /// Record a compilation step.
    pub fn record_step(
        &mut self,
        phase: impl Into<String>,
        description: impl Into<String>,
        ctx: &CompilerContext,
        graph: &EinsumGraph,
    ) {
        if self.enabled {
            if let Some(ref mut trace) = self.trace {
                trace.add_step(phase, description, ctx, graph);
            }
        }
    }

    /// Record an error.
    pub fn record_error(&mut self, error: impl Into<String>) {
        if self.enabled {
            if let Some(ref mut trace) = self.trace {
                trace.add_error(error);
            }
        }
    }

    /// Finish tracing and return the trace.
    pub fn finish(&mut self, graph: &EinsumGraph) -> Option<CompilationTrace> {
        if !self.enabled {
            return None;
        }

        if let Some(ref mut trace) = self.trace {
            trace.set_final_graph(graph);

            if let Some(start) = self.start_time {
                let duration = start.elapsed();
                trace.set_duration(duration.as_secs_f64() * 1000.0);
            }
        }

        self.trace.take()
    }
}

/// Print the compiler context state for debugging.
pub fn print_context_state(ctx: &CompilerContext, label: &str) {
    println!("\n=== Context State: {} ===", label);
    println!("Domains: {}", ctx.domains.len());
    for (name, info) in &ctx.domains {
        println!("  - {} (cardinality: {})", name, info.cardinality);
    }

    println!("Var->Domain bindings: {}", ctx.var_to_domain.len());
    for (var, domain) in &ctx.var_to_domain {
        println!("  - {} -> {}", var, domain);
    }

    println!("Var->Axis assignments: {}", ctx.var_to_axis.len());
    for (var, axis) in &ctx.var_to_axis {
        println!("  - {} -> axis '{}'", var, axis);
    }

    println!("Config: {:?}", ctx.config.and_strategy);
    println!("========================\n");
}

/// Print the graph state for debugging.
pub fn print_graph_state(graph: &EinsumGraph, label: &str) {
    println!("\n=== Graph State: {} ===", label);
    println!("Tensors: {}", graph.tensors.len());
    for (i, tensor) in graph.tensors.iter().enumerate() {
        println!("  [{:3}] {}", i, tensor);
    }

    println!("Nodes: {}", graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        println!("  [{:3}] {:?}", i, node.op);
        println!(
            "        inputs: {:?}, outputs: {:?}",
            node.inputs, node.outputs
        );
    }

    println!("Inputs: {:?}", graph.inputs);
    println!("Outputs: {:?}", graph.outputs);
    println!("========================\n");
}

/// Diff two graphs and print the differences.
pub fn print_graph_diff(before: &EinsumGraph, after: &EinsumGraph, label: &str) {
    println!("\n=== Graph Diff: {} ===", label);

    let tensor_diff = after.tensors.len() as i32 - before.tensors.len() as i32;
    let node_diff = after.nodes.len() as i32 - before.nodes.len() as i32;

    println!(
        "Tensors: {} -> {} ({:+})",
        before.tensors.len(),
        after.tensors.len(),
        tensor_diff
    );
    println!(
        "Nodes: {} -> {} ({:+})",
        before.nodes.len(),
        after.nodes.len(),
        node_diff
    );

    if tensor_diff > 0 {
        println!("New tensors:");
        for tensor in &after.tensors[before.tensors.len()..] {
            println!("  + {}", tensor);
        }
    }

    if node_diff > 0 {
        println!("New nodes:");
        for (i, node) in after.nodes[before.nodes.len()..].iter().enumerate() {
            let idx = before.nodes.len() + i;
            println!("  + [{:3}] {:?}", idx, node.op);
        }
    }

    println!("========================\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompilerContext;
    use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr, Term};

    #[test]
    fn test_compilation_trace_creation() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let trace = CompilationTrace::new(&expr);
        assert_eq!(trace.steps.len(), 0);
        assert_eq!(trace.errors.len(), 0);
        assert!(trace.final_graph.is_none());
    }

    #[test]
    fn test_add_compilation_step() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let mut trace = CompilationTrace::new(&expr);
        let ctx = CompilerContext::new();
        let graph = EinsumGraph::new();

        trace.add_step("Parse", "Parsed expression", &ctx, &graph);

        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.steps[0].phase, "Parse");
        assert_eq!(trace.steps[0].description, "Parsed expression");
    }

    #[test]
    fn test_compilation_tracer_disabled() {
        let mut tracer = CompilationTracer::new(false);

        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        tracer.start(&expr);

        let ctx = CompilerContext::new();
        let graph = EinsumGraph::new();

        tracer.record_step("Test", "Description", &ctx, &graph);

        let result = tracer.finish(&graph);
        assert!(result.is_none());
    }

    #[test]
    fn test_compilation_tracer_enabled() {
        let mut tracer = CompilationTracer::new(true);

        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        tracer.start(&expr);

        let ctx = CompilerContext::new();
        let graph = EinsumGraph::new();

        tracer.record_step("Phase1", "First step", &ctx, &graph);
        tracer.record_step("Phase2", "Second step", &ctx, &graph);

        let result = tracer.finish(&graph);
        assert!(result.is_some());

        let trace = result.unwrap();
        assert_eq!(trace.steps.len(), 2);
        assert!(trace.duration_ms.is_some());
    }

    #[test]
    fn test_print_context_state() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D1".to_string(), 10);
        // bind_var requires a domain name, not an axis number
        let _ = ctx.bind_var("x", "D1");

        // Should not panic
        print_context_state(&ctx, "Test");
    }

    #[test]
    fn test_print_graph_state() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());

        graph
            .add_node(EinsumNode::elem_unary("relu", t0, t1))
            .unwrap();

        // Should not panic
        print_graph_state(&graph, "Test");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world this is long", 10), "hello worl...");
    }
}
