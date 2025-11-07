//! DOT format export for graph visualization.
//!
//! This module provides utilities to export `EinsumGraph` to DOT format
//! for visualization with Graphviz and similar tools.
//!
//! # Example
//!
//! ```
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//!
//! let mut graph = EinsumGraph::new();
//! let t0 = graph.add_tensor("input".to_string());
//! let t1 = graph.add_tensor("output".to_string());
//! let node = EinsumNode::elem_unary("relu", t0, t1);
//! graph.add_node(node).unwrap();
//!
//! let dot = tensorlogic_ir::export_to_dot(&graph);
//! println!("{}", dot);
//! ```

use crate::graph::{EinsumGraph, EinsumNode, OpType};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;

/// Export an `EinsumGraph` to DOT format.
///
/// The resulting DOT string can be rendered with Graphviz:
/// ```bash
/// echo "..." | dot -Tpng > graph.png
/// echo "..." | dot -Tsvg > graph.svg
/// ```
///
/// # Layout Options
///
/// The generated DOT uses the following attributes:
/// - **Tensor nodes**: Boxes with blue color
/// - **Operation nodes**: Ellipses with green color
/// - **Edges**: Show data flow from inputs to operations to outputs
///
/// # Example
///
/// ```
/// use tensorlogic_ir::{EinsumGraph, EinsumNode, export_to_dot};
///
/// let mut graph = EinsumGraph::new();
/// let input = graph.add_tensor("x".to_string());
/// let output = graph.add_tensor("y".to_string());
///
/// let node = EinsumNode::elem_unary("relu", input, output);
/// graph.add_node(node).unwrap();
///
/// let dot = export_to_dot(&graph);
/// assert!(dot.contains("digraph"));
/// assert!(dot.contains("relu"));
/// ```
pub fn export_to_dot(graph: &EinsumGraph) -> String {
    let mut output = String::new();
    export_to_dot_writer(graph, &mut output).expect("String write should not fail");
    output
}

/// Export an `EinsumGraph` to DOT format with custom options.
///
/// # Options
///
/// - `show_tensor_ids`: Show tensor indices in labels
/// - `show_node_ids`: Show node indices in labels
/// - `show_metadata`: Include metadata in node labels
/// - `cluster_by_operation`: Group operations by type
/// - `horizontal_layout`: Use left-to-right layout instead of top-to-bottom
///
/// # Example
///
/// ```
/// use tensorlogic_ir::{EinsumGraph, EinsumNode, DotExportOptions, export_to_dot_with_options};
///
/// let mut graph = EinsumGraph::new();
/// let t0 = graph.add_tensor("input".to_string());
/// let t1 = graph.add_tensor("output".to_string());
/// let node = EinsumNode::elem_unary("sigmoid", t0, t1);
/// graph.add_node(node).unwrap();
///
/// let options = DotExportOptions {
///     show_tensor_ids: true,
///     show_node_ids: true,
///     horizontal_layout: true,
///     ..Default::default()
/// };
///
/// let dot = export_to_dot_with_options(&graph, &options);
/// assert!(dot.contains("rankdir=LR"));
/// ```
pub fn export_to_dot_with_options(graph: &EinsumGraph, options: &DotExportOptions) -> String {
    let mut output = String::new();
    export_to_dot_writer_with_options(graph, &mut output, options)
        .expect("String write should not fail");
    output
}

/// Options for DOT export customization.
#[derive(Debug, Clone, Default)]
pub struct DotExportOptions {
    /// Show tensor indices in labels (e.g., "tensor_0 \[0\]")
    pub show_tensor_ids: bool,
    /// Show node indices in labels (e.g., "op_0")
    pub show_node_ids: bool,
    /// Include metadata in node labels
    pub show_metadata: bool,
    /// Group operations by type (einsum, elem_unary, elem_binary, reduce)
    pub cluster_by_operation: bool,
    /// Use horizontal (left-to-right) layout instead of vertical
    pub horizontal_layout: bool,
    /// Include tensor shapes in labels (if available)
    pub show_shapes: bool,
    /// Highlight specific tensors (by name or index)
    pub highlight_tensors: Vec<String>,
    /// Highlight specific operations (by index)
    pub highlight_nodes: Vec<usize>,
}

/// Export to DOT format writing to a generic writer.
pub fn export_to_dot_writer<W: FmtWrite>(graph: &EinsumGraph, writer: &mut W) -> std::fmt::Result {
    let options = DotExportOptions::default();
    export_to_dot_writer_with_options(graph, writer, &options)
}

/// Export to DOT format with options, writing to a generic writer.
pub fn export_to_dot_writer_with_options<W: FmtWrite>(
    graph: &EinsumGraph,
    writer: &mut W,
    options: &DotExportOptions,
) -> std::fmt::Result {
    writeln!(writer, "digraph EinsumGraph {{")?;

    // Graph attributes
    writeln!(writer, "  // Graph styling")?;
    writeln!(writer, "  graph [fontname=\"Helvetica\", fontsize=10];")?;
    writeln!(writer, "  node [fontname=\"Helvetica\", fontsize=10];")?;
    writeln!(writer, "  edge [fontname=\"Helvetica\", fontsize=9];")?;

    if options.horizontal_layout {
        writeln!(writer, "  rankdir=LR;")?;
    }

    writeln!(writer)?;

    // Group operations by type if requested
    let mut op_clusters: HashMap<String, Vec<usize>> = HashMap::new();
    if options.cluster_by_operation {
        for (idx, node) in graph.nodes.iter().enumerate() {
            let cluster_name = match &node.op {
                OpType::Einsum { .. } => "einsum",
                OpType::ElemUnary { .. } => "elem_unary",
                OpType::ElemBinary { .. } => "elem_binary",
                OpType::Reduce { .. } => "reduce",
            };
            op_clusters
                .entry(cluster_name.to_string())
                .or_default()
                .push(idx);
        }
    }

    // Collect input and output tensors
    let mut used_tensors = HashSet::new();
    for node in &graph.nodes {
        for &input in &node.inputs {
            used_tensors.insert(input);
        }
        for &output in &node.outputs {
            used_tensors.insert(output);
        }
    }

    // Write tensor nodes
    writeln!(writer, "  // Tensor nodes")?;
    for (idx, tensor_name) in graph.tensors.iter().enumerate() {
        if !used_tensors.contains(&idx) && !graph.inputs.contains(&idx) {
            continue; // Skip unused tensors
        }

        let label = if options.show_tensor_ids {
            format!("{} [{}]", escape_label(tensor_name), idx)
        } else {
            escape_label(tensor_name)
        };

        let is_input = graph.inputs.contains(&idx);
        let is_output = graph.outputs.contains(&idx);
        let is_highlighted = options.highlight_tensors.contains(tensor_name)
            || options
                .highlight_tensors
                .contains(&format!("tensor_{}", idx));

        let color = if is_highlighted {
            "red"
        } else if is_input && is_output {
            "purple"
        } else if is_input {
            "lightblue"
        } else if is_output {
            "lightgreen"
        } else {
            "lightyellow"
        };

        writeln!(
            writer,
            "  tensor_{} [label=\"{}\", shape=box, style=filled, fillcolor={}];",
            idx, label, color
        )?;
    }

    writeln!(writer)?;

    // Write operation nodes, possibly clustered
    if options.cluster_by_operation && !op_clusters.is_empty() {
        for (cluster_name, node_indices) in &op_clusters {
            writeln!(
                writer,
                "  subgraph cluster_{} {{",
                cluster_name.replace('.', "_")
            )?;
            writeln!(writer, "    label=\"{}\";", cluster_name)?;
            writeln!(writer, "    style=dashed;")?;

            for &node_idx in node_indices {
                write_operation_node(writer, &graph.nodes[node_idx], node_idx, options)?;
            }

            writeln!(writer, "  }}")?;
            writeln!(writer)?;
        }
    } else {
        writeln!(writer, "  // Operation nodes")?;
        for (idx, node) in graph.nodes.iter().enumerate() {
            write_operation_node(writer, node, idx, options)?;
        }
        writeln!(writer)?;
    }

    // Write edges
    writeln!(writer, "  // Data flow edges")?;
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        // Input edges
        for &input_tensor in &node.inputs {
            writeln!(writer, "  tensor_{} -> op_{};", input_tensor, node_idx)?;
        }

        // Output edges
        for &output_tensor in &node.outputs {
            writeln!(writer, "  op_{} -> tensor_{};", node_idx, output_tensor)?;
        }
    }

    writeln!(writer, "}}")?;

    Ok(())
}

/// Write a single operation node to the DOT output.
fn write_operation_node<W: FmtWrite>(
    writer: &mut W,
    node: &EinsumNode,
    idx: usize,
    options: &DotExportOptions,
) -> std::fmt::Result {
    let (op_type, op_label) = match &node.op {
        OpType::Einsum { spec } => ("einsum", format!("einsum\\n{}", escape_label(spec))),
        OpType::ElemUnary { op } => ("elem_unary", format!("{}(·)", escape_label(op))),
        OpType::ElemBinary { op } => ("elem_binary", format!("{}(·,·)", escape_label(op))),
        OpType::Reduce { op, axes } => ("reduce", format!("{}(axes={:?})", escape_label(op), axes)),
    };

    let label = if options.show_node_ids {
        format!("{}\\n[op_{}]", op_label, idx)
    } else {
        op_label
    };

    let is_highlighted = options.highlight_nodes.contains(&idx);
    let color = if is_highlighted {
        "orange"
    } else {
        match op_type {
            "einsum" => "lightcyan",
            "elem_unary" => "lightgreen",
            "elem_binary" => "lightyellow",
            "reduce" => "lightpink",
            _ => "white",
        }
    };

    writeln!(
        writer,
        "  op_{} [label=\"{}\", shape=ellipse, style=filled, fillcolor={}];",
        idx, label, color
    )?;

    Ok(())
}

/// Escape special characters in DOT labels.
fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EinsumGraph, EinsumNode};

    #[test]
    fn test_export_empty_graph() {
        let graph = EinsumGraph::new();
        let dot = export_to_dot(&graph);
        assert!(dot.contains("digraph EinsumGraph"));
    }

    #[test]
    fn test_export_simple_operation() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());

        let node = EinsumNode::elem_unary("relu", t0, t1);
        graph.add_node(node).unwrap();

        let dot = export_to_dot(&graph);
        assert!(dot.contains("relu"));
        assert!(dot.contains("tensor_0"));
        assert!(dot.contains("tensor_1"));
        assert!(dot.contains("op_0"));
    }

    #[test]
    fn test_export_with_einsum() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("A".to_string());
        let t1 = graph.add_tensor("B".to_string());
        let t2 = graph.add_tensor("C".to_string());

        let node = EinsumNode::einsum("ij,jk->ik", vec![t0, t1], vec![t2]);
        graph.add_node(node).unwrap();

        let dot = export_to_dot(&graph);
        assert!(dot.contains("einsum"));
        assert!(dot.contains("ij,jk->ik"));
    }

    #[test]
    fn test_export_with_options() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("x".to_string());
        let t1 = graph.add_tensor("y".to_string());

        let node = EinsumNode::elem_unary("sigmoid", t0, t1);
        graph.add_node(node).unwrap();

        let options = DotExportOptions {
            show_tensor_ids: true,
            show_node_ids: true,
            horizontal_layout: true,
            ..Default::default()
        };

        let dot = export_to_dot_with_options(&graph, &options);
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("[0]")); // Tensor ID
        assert!(dot.contains("[op_0]")); // Node ID
    }

    #[test]
    fn test_export_with_clustering() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("a".to_string());
        let t1 = graph.add_tensor("b".to_string());
        let t2 = graph.add_tensor("c".to_string());
        let t3 = graph.add_tensor("d".to_string());

        graph
            .add_node(EinsumNode::elem_unary("relu", t0, t1))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("sigmoid", t1, t2))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_binary("add", t2, t0, t3))
            .unwrap();

        let options = DotExportOptions {
            cluster_by_operation: true,
            ..Default::default()
        };

        let dot = export_to_dot_with_options(&graph, &options);
        assert!(dot.contains("subgraph cluster_elem_unary"));
        assert!(dot.contains("subgraph cluster_elem_binary"));
    }

    #[test]
    fn test_export_with_highlights() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("hidden".to_string());
        let t2 = graph.add_tensor("output".to_string());

        graph
            .add_node(EinsumNode::elem_unary("relu", t0, t1))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("softmax", t1, t2))
            .unwrap();

        let options = DotExportOptions {
            highlight_tensors: vec!["output".to_string()],
            highlight_nodes: vec![0],
            ..Default::default()
        };

        let dot = export_to_dot_with_options(&graph, &options);
        assert!(dot.contains("red")); // Highlighted tensor
        assert!(dot.contains("orange")); // Highlighted operation
    }

    #[test]
    fn test_label_escaping() {
        assert_eq!(escape_label("hello\"world"), "hello\\\"world");
        assert_eq!(escape_label("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_label("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_complex_graph_export() {
        let mut graph = EinsumGraph::new();

        // Build a more complex graph: (a + b) * c
        let a = graph.add_tensor("a".to_string());
        let b = graph.add_tensor("b".to_string());
        let c = graph.add_tensor("c".to_string());
        let sum = graph.add_tensor("sum".to_string());
        let result = graph.add_tensor("result".to_string());

        graph.inputs = vec![a, b, c];
        graph.outputs = vec![result];

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, sum))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_binary("multiply", sum, c, result))
            .unwrap();

        let dot = export_to_dot(&graph);

        // Verify structure
        assert!(dot.contains("tensor_0")); // a
        assert!(dot.contains("tensor_1")); // b
        assert!(dot.contains("tensor_2")); // c
        assert!(dot.contains("tensor_3")); // sum
        assert!(dot.contains("tensor_4")); // result
        assert!(dot.contains("op_0")); // add
        assert!(dot.contains("op_1")); // multiply

        // Verify edges
        assert!(dot.contains("tensor_0 -> op_0")); // a -> add
        assert!(dot.contains("tensor_1 -> op_0")); // b -> add
        assert!(dot.contains("op_0 -> tensor_3")); // add -> sum
        assert!(dot.contains("tensor_3 -> op_1")); // sum -> multiply
        assert!(dot.contains("tensor_2 -> op_1")); // c -> multiply
        assert!(dot.contains("op_1 -> tensor_4")); // multiply -> result
    }
}
