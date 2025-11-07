//! Visualization utilities for execution analysis and debugging.
//!
//! This module provides tools for visualizing:
//! - Execution timelines
//! - Computation graphs
//! - Performance data
//!
//! Supports multiple output formats:
//! - ASCII art for terminal display
//! - GraphViz DOT format
//! - JSON for custom rendering

use crate::debug::{ExecutionTrace, TensorStats};
use crate::profiling::ProfileData;
use tensorlogic_ir::EinsumGraph;

/// Visualization format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// ASCII art for terminal display
    Ascii,
    /// GraphViz DOT format
    Dot,
    /// JSON format
    Json,
    /// HTML format
    Html,
}

/// Timeline visualization configuration.
#[derive(Debug, Clone)]
pub struct TimelineConfig {
    /// Width in characters (for ASCII output)
    pub width: usize,
    /// Show operation names
    pub show_names: bool,
    /// Show timing information
    pub show_timing: bool,
    /// Group by operation type
    pub group_by_type: bool,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            width: 80,
            show_names: true,
            show_timing: true,
            group_by_type: false,
        }
    }
}

/// Graph visualization configuration.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Show tensor shapes
    pub show_shapes: bool,
    /// Show operation types
    pub show_op_types: bool,
    /// Highlight critical path
    pub highlight_critical_path: bool,
    /// Vertical or horizontal layout
    pub vertical_layout: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            show_shapes: true,
            show_op_types: true,
            highlight_critical_path: false,
            vertical_layout: true,
        }
    }
}

/// Timeline visualizer.
pub struct TimelineVisualizer {
    config: TimelineConfig,
}

impl TimelineVisualizer {
    /// Create a new timeline visualizer.
    pub fn new(config: TimelineConfig) -> Self {
        Self { config }
    }

    /// Visualize an execution trace as ASCII timeline.
    pub fn visualize_trace(&self, trace: &ExecutionTrace) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!(
            "Execution Timeline ({:.2}ms total)\n",
            trace.total_duration_ms()
        ));
        output.push_str(&"=".repeat(self.config.width));
        output.push('\n');

        if trace.entries().is_empty() {
            output.push_str("No operations recorded\n");
            return output;
        }

        // Find time bounds
        let start_time = trace.entries()[0].start_time;
        let total_duration = trace.total_duration();

        // Draw timeline
        for entry in trace.entries() {
            let elapsed = entry.start_time.duration_since(start_time);
            let duration = entry.duration;

            // Calculate bar position and width
            let start_pos = (elapsed.as_secs_f64() / total_duration.as_secs_f64()
                * self.config.width as f64) as usize;
            let bar_width = ((duration.as_secs_f64() / total_duration.as_secs_f64()
                * self.config.width as f64) as usize)
                .max(1);

            // Operation name
            if self.config.show_names {
                output.push_str(&format!("Node {}: {} ", entry.node_id, entry.operation));
            }

            if self.config.show_timing {
                output.push_str(&format!("({:.2}ms)\n", entry.duration_ms()));
            } else {
                output.push('\n');
            }

            // Timeline bar
            output.push_str(&" ".repeat(start_pos));
            output.push_str(&"█".repeat(bar_width));
            output.push('\n');
        }

        output.push_str(&"=".repeat(self.config.width));
        output.push('\n');

        output
    }

    /// Visualize profile data as text report.
    pub fn visualize_profile(&self, profile: &ProfileData) -> String {
        let mut output = String::new();

        output.push_str("Performance Profile\n");
        output.push_str(&"=".repeat(self.config.width));
        output.push('\n');

        // Sort operations by total time
        let mut ops: Vec<_> = profile.op_profiles.iter().collect();
        ops.sort_by(|(_, a), (_, b)| {
            let a_total_ms = a.avg_time.as_secs_f64() * 1000.0 * a.count as f64;
            let b_total_ms = b.avg_time.as_secs_f64() * 1000.0 * b.count as f64;
            b_total_ms.partial_cmp(&a_total_ms).unwrap()
        });

        // Header
        output.push_str(&format!(
            "{:<30} {:>10} {:>10} {:>15}\n",
            "Operation", "Count", "Avg (ms)", "Total (ms)"
        ));
        output.push_str(&"-".repeat(self.config.width));
        output.push('\n');

        // Operations
        for (name, stats) in ops {
            let avg_time_ms = stats.avg_time.as_secs_f64() * 1000.0;
            let total_time_ms = avg_time_ms * stats.count as f64;
            output.push_str(&format!(
                "{:<30} {:>10} {:>10.2} {:>15.2}\n",
                name, stats.count, avg_time_ms, total_time_ms
            ));
        }

        output.push_str(&"=".repeat(self.config.width));
        output.push('\n');

        output
    }
}

/// Graph visualizer.
pub struct GraphVisualizer {
    config: GraphConfig,
}

impl GraphVisualizer {
    /// Create a new graph visualizer.
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }

    /// Visualize a computation graph as ASCII art.
    pub fn visualize_ascii(&self, graph: &EinsumGraph) -> String {
        let mut output = String::new();

        output.push_str("Computation Graph\n");
        output.push_str("=================\n\n");

        if graph.nodes.is_empty() {
            output.push_str("Empty graph\n");
            return output;
        }

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Node representation
            output.push_str(&format!("Node {}:\n", node_idx));

            // Operation type
            if self.config.show_op_types {
                output.push_str(&format!("  Op: {:?}\n", node.op));
            }

            // Inputs
            if !node.inputs.is_empty() {
                output.push_str("  Inputs: ");
                for (i, input_id) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    output.push_str(&format!("{}", input_id));
                }
                output.push('\n');
            }

            output.push('\n');
        }

        output
    }

    /// Generate GraphViz DOT format.
    pub fn visualize_dot(&self, graph: &EinsumGraph) -> String {
        let mut output = String::new();

        output.push_str("digraph ComputationGraph {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box, style=rounded];\n\n");

        // Nodes
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let label = format!("Node {}\\n{:?}", node_idx, node.op);
            output.push_str(&format!("  n{} [label=\"{}\"];\n", node_idx, label));
        }

        output.push('\n');

        // Edges
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for input_id in &node.inputs {
                output.push_str(&format!("  n{} -> n{};\n", input_id, node_idx));
            }
        }

        output.push_str("}\n");

        output
    }

    /// Generate JSON representation.
    pub fn visualize_json(&self, graph: &EinsumGraph) -> String {
        let mut output = String::new();

        output.push_str("{\n");
        output.push_str("  \"nodes\": [\n");

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            if node_idx > 0 {
                output.push_str(",\n");
            }
            output.push_str("    {\n");
            output.push_str(&format!("      \"id\": {},\n", node_idx));
            output.push_str(&format!("      \"op\": \"{:?}\",\n", node.op));
            output.push_str("      \"inputs\": [");

            for (j, input_id) in node.inputs.iter().enumerate() {
                if j > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("{}", input_id));
            }

            output.push_str("]\n");
            output.push_str("    }");
        }

        output.push_str("\n  ]\n");
        output.push_str("}\n");

        output
    }
}

/// Tensor statistics visualizer.
pub struct TensorStatsVisualizer;

impl TensorStatsVisualizer {
    /// Visualize tensor statistics as a text report.
    pub fn visualize(&self, stats: &TensorStats) -> String {
        format!("{}", stats)
    }

    /// Visualize multiple tensor statistics as a table.
    pub fn visualize_table(&self, stats: &[TensorStats]) -> String {
        let mut output = String::new();

        output.push_str("Tensor Statistics\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        if stats.is_empty() {
            output.push_str("No tensors recorded\n");
            return output;
        }

        // Header
        output.push_str(&format!(
            "{:<8} {:<20} {:<15} {:>10} {:>10}\n",
            "ID", "Shape", "DType", "NaNs", "Infs"
        ));
        output.push_str(&"-".repeat(80));
        output.push('\n');

        // Rows
        for stat in stats {
            let shape_str = format!("{:?}", stat.shape);
            let nans = stat.num_nans.unwrap_or(0);
            let infs = stat.num_infs.unwrap_or(0);

            output.push_str(&format!(
                "{:<8} {:<20} {:<15} {:>10} {:>10}\n",
                stat.tensor_id, shape_str, stat.dtype, nans, infs
            ));

            // Highlight issues
            if stat.has_numerical_issues() {
                output.push_str("         ⚠️  Numerical issues detected!\n");
            }
        }

        output.push_str(&"=".repeat(80));
        output.push('\n');

        output
    }

    /// Generate histogram of tensor values (ASCII).
    pub fn histogram(&self, values: &[f64], bins: usize) -> String {
        let mut output = String::new();

        if values.is_empty() {
            return "No values\n".to_string();
        }

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        if range == 0.0 {
            return format!("All values are {:.6}\n", min);
        }

        // Create bins
        let mut counts = vec![0; bins];
        for &value in values {
            let bin = ((value - min) / range * bins as f64) as usize;
            let bin = bin.min(bins - 1);
            counts[bin] += 1;
        }

        let max_count = *counts.iter().max().unwrap();

        // Draw histogram
        output.push_str("Value Distribution\n");
        output.push_str(&"=".repeat(50));
        output.push('\n');

        for (i, &count) in counts.iter().enumerate() {
            let bin_start = min + (i as f64 / bins as f64) * range;
            let bin_end = min + ((i + 1) as f64 / bins as f64) * range;
            let bar_width = if max_count > 0 {
                (count as f64 / max_count as f64 * 40.0) as usize
            } else {
                0
            };

            output.push_str(&format!(
                "[{:>8.2}, {:>8.2}): {} ({})\n",
                bin_start,
                bin_end,
                "█".repeat(bar_width),
                count
            ));
        }

        output.push_str(&"=".repeat(50));
        output.push('\n');

        output
    }
}

/// Export formats for external visualization tools.
pub struct ExportFormat;

impl ExportFormat {
    /// Export trace to JSON for custom visualization.
    pub fn trace_to_json(trace: &ExecutionTrace) -> String {
        let mut output = String::new();

        output.push_str("{\n");
        output.push_str(&format!(
            "  \"total_duration_ms\": {},\n",
            trace.total_duration_ms()
        ));
        output.push_str("  \"entries\": [\n");

        for (i, entry) in trace.entries().iter().enumerate() {
            if i > 0 {
                output.push_str(",\n");
            }
            output.push_str("    {\n");
            output.push_str(&format!("      \"entry_id\": {},\n", entry.entry_id));
            output.push_str(&format!("      \"node_id\": {},\n", entry.node_id));
            output.push_str(&format!("      \"operation\": \"{}\",\n", entry.operation));
            output.push_str(&format!(
                "      \"duration_ms\": {},\n",
                entry.duration_ms()
            ));
            output.push_str(&format!("      \"input_ids\": {:?},\n", entry.input_ids));
            output.push_str(&format!("      \"output_ids\": {:?}\n", entry.output_ids));
            output.push_str("    }");
        }

        output.push_str("\n  ]\n");
        output.push_str("}\n");

        output
    }

    /// Export graph to GraphML format.
    pub fn graph_to_graphml(graph: &EinsumGraph) -> String {
        let mut output = String::new();

        output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        output.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
        output.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

        // Nodes
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            output.push_str(&format!("    <node id=\"n{}\">\n", node_idx));
            output.push_str(&format!(
                "      <data key=\"operation\">{:?}</data>\n",
                node.op
            ));
            output.push_str("    </node>\n");
        }

        // Edges
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for input_id in &node.inputs {
                output.push_str(&format!(
                    "    <edge source=\"n{}\" target=\"n{}\"/>\n",
                    input_id, node_idx
                ));
            }
        }

        output.push_str("  </graph>\n");
        output.push_str("</graphml>\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::{ExecutionTracer, TensorStats};
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn test_timeline_visualizer() {
        let mut tracer = ExecutionTracer::new();
        tracer.enable();
        tracer.start_trace(Some(1));

        let handle = tracer.record_operation_start(0, "einsum", vec![]);
        std::thread::sleep(Duration::from_millis(10));
        tracer.record_operation_end(handle, 0, "einsum", vec![], vec![1], HashMap::new());

        let trace = tracer.get_trace();
        let visualizer = TimelineVisualizer::new(TimelineConfig::default());
        let output = visualizer.visualize_trace(trace);

        assert!(output.contains("Execution Timeline"));
        assert!(output.contains("Node 0"));
        assert!(output.contains("einsum"));
    }

    #[test]
    fn test_graph_visualizer_ascii() {
        use tensorlogic_ir::EinsumNode;

        let graph = EinsumGraph {
            tensors: vec!["input".to_string(), "output".to_string()],
            nodes: vec![EinsumNode::new("ij->ij", vec![], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            tensor_metadata: HashMap::new(),
        };

        let visualizer = GraphVisualizer::new(GraphConfig::default());
        let output = visualizer.visualize_ascii(&graph);

        assert!(output.contains("Computation Graph"));
        assert!(output.contains("Node 0"));
    }

    #[test]
    fn test_graph_visualizer_dot() {
        use tensorlogic_ir::EinsumNode;

        let graph = EinsumGraph {
            tensors: vec!["input".to_string(), "output".to_string()],
            nodes: vec![EinsumNode::new("ij->ij", vec![], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            tensor_metadata: HashMap::new(),
        };

        let visualizer = GraphVisualizer::new(GraphConfig::default());
        let output = visualizer.visualize_dot(&graph);

        assert!(output.contains("digraph ComputationGraph"));
        assert!(output.contains("n0"));
    }

    #[test]
    fn test_graph_visualizer_json() {
        use tensorlogic_ir::EinsumNode;

        let graph = EinsumGraph {
            tensors: vec!["input".to_string(), "output".to_string()],
            nodes: vec![EinsumNode::new("ij->ij", vec![], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            tensor_metadata: HashMap::new(),
        };

        let visualizer = GraphVisualizer::new(GraphConfig::default());
        let output = visualizer.visualize_json(&graph);

        assert!(output.contains("\"nodes\""));
        assert!(output.contains("\"id\": 0"));
    }

    #[test]
    fn test_tensor_stats_visualizer() {
        let stats =
            TensorStats::new(0, vec![2, 3], "f64").with_statistics(0.0, 1.0, 0.5, 0.25, 0, 0);

        let visualizer = TensorStatsVisualizer;
        let output = visualizer.visualize(&stats);

        assert!(output.contains("Tensor 0"));
        assert!(output.contains("f64"));
    }

    #[test]
    fn test_tensor_stats_table() {
        let stats = vec![
            TensorStats::new(0, vec![2, 3], "f64"),
            TensorStats::new(1, vec![4, 5], "f64"),
        ];

        let visualizer = TensorStatsVisualizer;
        let output = visualizer.visualize_table(&stats);

        assert!(output.contains("Tensor Statistics"));
        assert!(output.contains("ID"));
        assert!(output.contains("Shape"));
    }

    #[test]
    fn test_histogram() {
        let values = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let visualizer = TensorStatsVisualizer;
        let output = visualizer.histogram(&values, 5);

        assert!(output.contains("Value Distribution"));
        assert!(output.contains("█"));
    }

    #[test]
    fn test_export_trace_to_json() {
        let mut tracer = ExecutionTracer::new();
        tracer.enable();
        tracer.start_trace(Some(1));

        let handle = tracer.record_operation_start(0, "einsum", vec![]);
        tracer.record_operation_end(handle, 0, "einsum", vec![], vec![1], HashMap::new());

        let trace = tracer.get_trace();
        let json = ExportFormat::trace_to_json(trace);

        assert!(json.contains("total_duration_ms"));
        assert!(json.contains("entries"));
        assert!(json.contains("\"operation\": \"einsum\""));
    }

    #[test]
    fn test_export_graph_to_graphml() {
        use tensorlogic_ir::EinsumNode;

        let graph = EinsumGraph {
            tensors: vec!["input".to_string(), "output".to_string()],
            nodes: vec![EinsumNode::new("ij->ij", vec![], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            tensor_metadata: HashMap::new(),
        };

        let graphml = ExportFormat::graph_to_graphml(&graph);

        assert!(graphml.contains("<?xml"));
        assert!(graphml.contains("<graphml"));
        assert!(graphml.contains("<node id=\"n0\""));
    }
}
