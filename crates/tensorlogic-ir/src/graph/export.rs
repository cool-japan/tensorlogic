//! Graph export to standard ML formats.
//!
//! This module provides export functionality for EinsumGraph to various
//! machine learning interchange formats:
//! - ONNX (Open Neural Network Exchange) text representation
//! - TorchScript text representation
//! - Textual IR representations
//!
//! # Examples
//!
//! ```no_run
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//! use tensorlogic_ir::{export_to_onnx_text, export_to_torchscript_text};
//!
//! let mut graph = EinsumGraph::new();
//! let a = graph.add_tensor("A");
//! let b = graph.add_tensor("B");
//! let c = graph.add_tensor("C");
//!
//! graph.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c])).unwrap();
//! graph.add_output(c).unwrap();
//!
//! // Export to ONNX text format
//! let onnx_text = export_to_onnx_text(&graph).unwrap();
//! println!("{}", onnx_text);
//!
//! // Export to TorchScript text format
//! let torchscript = export_to_torchscript_text(&graph).unwrap();
//! println!("{}", torchscript);
//! ```

use crate::error::IrError;
use crate::graph::{EinsumGraph, OpType};
use std::fmt::Write as FmtWrite;

/// Export options for ONNX format.
#[derive(Clone, Debug)]
pub struct OnnxExportOptions {
    /// ONNX opset version to target (default: 13)
    pub opset_version: i64,
    /// Include metadata in export
    pub include_metadata: bool,
    /// Producer name
    pub producer_name: String,
    /// Model version
    pub model_version: i64,
}

impl Default for OnnxExportOptions {
    fn default() -> Self {
        Self {
            opset_version: 13,
            include_metadata: true,
            producer_name: "TensorLogic".to_string(),
            model_version: 1,
        }
    }
}

/// Export options for TorchScript format.
#[derive(Clone, Debug)]
pub struct TorchScriptExportOptions {
    /// Include type annotations
    pub include_types: bool,
    /// Include comments
    pub include_comments: bool,
    /// Optimize for inference (freeze parameters)
    pub optimize_for_inference: bool,
}

impl Default for TorchScriptExportOptions {
    fn default() -> Self {
        Self {
            include_types: true,
            include_comments: true,
            optimize_for_inference: false,
        }
    }
}

/// Export EinsumGraph to ONNX text representation.
///
/// This creates a textual representation of the ONNX model that describes
/// the computation graph structure. This can be used for debugging or
/// converted to binary ONNX format using ONNX tools.
///
/// # Examples
///
/// ```no_run
/// use tensorlogic_ir::{EinsumGraph, EinsumNode};
/// use tensorlogic_ir::export_to_onnx_text;
///
/// let mut graph = EinsumGraph::new();
/// let x = graph.add_tensor("X");
/// let y = graph.add_tensor("Y");
/// let z = graph.add_tensor("Z");
///
/// graph.add_node(EinsumNode::elem_binary("add", x, y, z)).unwrap();
/// graph.add_output(z).unwrap();
///
/// let onnx = export_to_onnx_text(&graph).unwrap();
/// assert!(onnx.contains("ir_version"));
/// assert!(onnx.contains("Add"));
/// ```
pub fn export_to_onnx_text(graph: &EinsumGraph) -> Result<String, IrError> {
    export_to_onnx_text_with_options(graph, &OnnxExportOptions::default())
}

/// Export EinsumGraph to ONNX text representation with custom options.
pub fn export_to_onnx_text_with_options(
    graph: &EinsumGraph,
    options: &OnnxExportOptions,
) -> Result<String, IrError> {
    let mut output = String::new();

    // ONNX header
    writeln!(output, "# ONNX Model: TensorLogic Computation Graph")?;
    writeln!(output, "# Producer: {}", options.producer_name)?;
    writeln!(output, "# Model Version: {}", options.model_version)?;
    writeln!(output)?;
    writeln!(output, "ir_version: 7")?;
    writeln!(output, "opset_import {{")?;
    writeln!(output, "  domain: \"\"")?;
    writeln!(output, "  version: {}", options.opset_version)?;
    writeln!(output, "}}")?;
    writeln!(output)?;

    // Model graph
    writeln!(output, "graph {{")?;
    writeln!(output, "  name: \"tensorlogic_graph\"")?;
    writeln!(output)?;

    // Inputs
    writeln!(output, "  # Inputs")?;
    for &input_idx in &graph.inputs {
        let tensor_name = &graph.tensors[input_idx];
        writeln!(output, "  input {{")?;
        writeln!(output, "    name: \"{}\"", tensor_name)?;
        writeln!(output, "    type {{")?;
        writeln!(output, "      tensor_type {{")?;
        writeln!(output, "        elem_type: 1  # FLOAT")?;
        writeln!(output, "        shape {{")?;
        writeln!(output, "          dim {{ dim_param: \"batch\" }}")?;
        writeln!(output, "          dim {{ dim_param: \"dynamic\" }}")?;
        writeln!(output, "        }}")?;
        writeln!(output, "      }}")?;
        writeln!(output, "    }}")?;
        writeln!(output, "  }}")?;
    }
    writeln!(output)?;

    // Nodes (operations)
    writeln!(output, "  # Operations")?;
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        export_node_to_onnx(&mut output, node, node_idx, graph)?;
    }
    writeln!(output)?;

    // Outputs
    writeln!(output, "  # Outputs")?;
    for &output_idx in &graph.outputs {
        let tensor_name = &graph.tensors[output_idx];
        writeln!(output, "  output {{")?;
        writeln!(output, "    name: \"{}\"", tensor_name)?;
        writeln!(output, "    type {{")?;
        writeln!(output, "      tensor_type {{")?;
        writeln!(output, "        elem_type: 1  # FLOAT")?;
        writeln!(output, "        shape {{")?;
        writeln!(output, "          dim {{ dim_param: \"batch\" }}")?;
        writeln!(output, "          dim {{ dim_param: \"dynamic\" }}")?;
        writeln!(output, "        }}")?;
        writeln!(output, "      }}")?;
        writeln!(output, "    }}")?;
        writeln!(output, "  }}")?;
    }

    writeln!(output, "}}")?;

    Ok(output)
}

/// Helper to export a single node to ONNX format.
fn export_node_to_onnx(
    output: &mut String,
    node: &crate::graph::EinsumNode,
    node_idx: usize,
    graph: &EinsumGraph,
) -> Result<(), IrError> {
    writeln!(output, "  node {{")?;

    // Input tensors
    for &input_idx in &node.inputs {
        writeln!(output, "    input: \"{}\"", graph.tensors[input_idx])?;
    }

    // Output tensors
    for &output_idx in &node.outputs {
        writeln!(output, "    output: \"{}\"", graph.tensors[output_idx])?;
    }

    // Operation type
    let op_name = match &node.op {
        OpType::Einsum { spec } => {
            writeln!(output, "    op_type: \"Einsum\"")?;
            writeln!(output, "    attribute {{")?;
            writeln!(output, "      name: \"equation\"")?;
            writeln!(output, "      s: \"{}\"", spec)?;
            writeln!(output, "      type: STRING")?;
            writeln!(output, "    }}")?;
            "Einsum"
        }
        OpType::ElemBinary { op } => {
            let onnx_op = match op.as_str() {
                "add" => "Add",
                "sub" => "Sub",
                "mul" => "Mul",
                "div" => "Div",
                _ => "Unknown",
            };
            writeln!(output, "    op_type: \"{}\"", onnx_op)?;
            onnx_op
        }
        OpType::ElemUnary { op } => {
            let onnx_op = match op.as_str() {
                "neg" => "Neg",
                "exp" => "Exp",
                "log" => "Log",
                "relu" => "Relu",
                "sigmoid" => "Sigmoid",
                "tanh" => "Tanh",
                _ => "Unknown",
            };
            writeln!(output, "    op_type: \"{}\"", onnx_op)?;
            onnx_op
        }
        OpType::Reduce { op, axes } => {
            let onnx_op = match op.as_str() {
                "sum" => "ReduceSum",
                "max" => "ReduceMax",
                "min" => "ReduceMin",
                "mean" => "ReduceMean",
                "prod" => "ReduceProd",
                _ => "Unknown",
            };
            writeln!(output, "    op_type: \"{}\"", onnx_op)?;
            if !axes.is_empty() {
                writeln!(output, "    attribute {{")?;
                writeln!(output, "      name: \"axes\"")?;
                write!(output, "      ints: ")?;
                for (i, axis) in axes.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ")?;
                    }
                    write!(output, "{}", axis)?;
                }
                writeln!(output)?;
                writeln!(output, "      type: INTS")?;
                writeln!(output, "    }}")?;
            }
            onnx_op
        }
    };

    writeln!(output, "    name: \"node_{}\"", node_idx)?;
    writeln!(output, "    doc_string: \"{} operation\"", op_name)?;
    writeln!(output, "  }}")?;

    Ok(())
}

/// Export EinsumGraph to TorchScript text representation.
///
/// This creates a PyTorch TorchScript representation that can be loaded
/// and executed by PyTorch's JIT compiler.
///
/// # Examples
///
/// ```no_run
/// use tensorlogic_ir::{EinsumGraph, EinsumNode};
/// use tensorlogic_ir::export_to_torchscript_text;
///
/// let mut graph = EinsumGraph::new();
/// let x = graph.add_tensor("X");
/// let w = graph.add_tensor("W");
/// let y = graph.add_tensor("Y");
///
/// graph.add_node(EinsumNode::einsum("ij,jk->ik", vec![x, w], vec![y])).unwrap();
/// graph.add_output(y).unwrap();
///
/// let script = export_to_torchscript_text(&graph).unwrap();
/// assert!(script.contains("torch.einsum"));
/// ```
pub fn export_to_torchscript_text(graph: &EinsumGraph) -> Result<String, IrError> {
    export_to_torchscript_text_with_options(graph, &TorchScriptExportOptions::default())
}

/// Export EinsumGraph to TorchScript text representation with custom options.
pub fn export_to_torchscript_text_with_options(
    graph: &EinsumGraph,
    options: &TorchScriptExportOptions,
) -> Result<String, IrError> {
    let mut output = String::new();

    // Header
    if options.include_comments {
        writeln!(
            output,
            "# TorchScript representation of TensorLogic computation graph"
        )?;
        writeln!(output, "# Generated by TensorLogic IR")?;
        writeln!(output)?;
    }

    writeln!(output, "import torch")?;
    writeln!(output, "import torch.nn as nn")?;
    writeln!(output)?;

    // Module class
    writeln!(output, "class TensorLogicGraph(nn.Module):")?;
    writeln!(output, "    def __init__(self):")?;
    writeln!(output, "        super(TensorLogicGraph, self).__init__()")?;
    writeln!(output)?;

    // Forward method
    write!(output, "    def forward(self")?;

    // Input parameters
    for &input_idx in &graph.inputs {
        write!(output, ", {}", graph.tensors[input_idx])?;
    }
    writeln!(output, "):")?;

    if options.include_comments {
        writeln!(output, "        # Computation graph")?;
    }

    // Generate operations
    for node in &graph.nodes {
        export_node_to_torchscript(&mut output, node, graph, options)?;
    }

    // Return statement
    writeln!(output)?;
    write!(output, "        return ")?;
    if graph.outputs.len() == 1 {
        writeln!(output, "{}", graph.tensors[graph.outputs[0]])?;
    } else {
        write!(output, "(")?;
        for (i, &output_idx) in graph.outputs.iter().enumerate() {
            if i > 0 {
                write!(output, ", ")?;
            }
            write!(output, "{}", graph.tensors[output_idx])?;
        }
        writeln!(output, ")")?;
    }

    Ok(output)
}

/// Helper to export a single node to TorchScript format.
fn export_node_to_torchscript(
    output: &mut String,
    node: &crate::graph::EinsumNode,
    graph: &EinsumGraph,
    options: &TorchScriptExportOptions,
) -> Result<(), IrError> {
    let output_tensor = graph.tensors[node.outputs[0]].clone();

    match &node.op {
        OpType::Einsum { spec } => {
            write!(
                output,
                "        {} = torch.einsum('{}', ",
                output_tensor, spec
            )?;
            for (i, &input_idx) in node.inputs.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ")?;
                }
                write!(output, "{}", graph.tensors[input_idx])?;
            }
            writeln!(output, ")")?;
        }
        OpType::ElemBinary { op } => {
            let input_tensors = &node.inputs;
            let torch_op = match op.as_str() {
                "add" => "torch.add",
                "sub" => "torch.sub",
                "mul" => "torch.mul",
                "div" => "torch.div",
                _ => "torch.unknown",
            };

            if options.include_comments {
                writeln!(output, "        # Element-wise binary operation: {}", op)?;
            }

            writeln!(
                output,
                "        {} = {}({}, {})",
                output_tensor,
                torch_op,
                graph.tensors[input_tensors[0]],
                graph.tensors[input_tensors[1]]
            )?;
        }
        OpType::ElemUnary { op } => {
            let input_tensor = graph.tensors[node.inputs[0]].clone();
            let torch_op = match op.as_str() {
                "neg" => "torch.neg",
                "exp" => "torch.exp",
                "log" => "torch.log",
                "relu" => "torch.relu",
                "sigmoid" => "torch.sigmoid",
                "tanh" => "torch.tanh",
                _ => "torch.unknown",
            };

            if options.include_comments {
                writeln!(output, "        # Element-wise unary operation: {}", op)?;
            }

            writeln!(
                output,
                "        {} = {}({})",
                output_tensor, torch_op, input_tensor
            )?;
        }
        OpType::Reduce { op, axes } => {
            let input_tensor = graph.tensors[node.inputs[0]].clone();
            let torch_op = match op.as_str() {
                "sum" => "sum",
                "max" => "max",
                "min" => "min",
                "mean" => "mean",
                "prod" => "prod",
                _ => "unknown",
            };

            if options.include_comments {
                writeln!(output, "        # Reduction operation: {}", op)?;
            }

            if axes.is_empty() {
                writeln!(
                    output,
                    "        {} = {}.{}()",
                    output_tensor, input_tensor, torch_op
                )?;
            } else {
                write!(
                    output,
                    "        {} = {}.{}(dim=[",
                    output_tensor, input_tensor, torch_op
                )?;
                for (i, axis) in axes.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ")?;
                    }
                    write!(output, "{}", axis)?;
                }
                writeln!(output, "])")?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EinsumGraph, EinsumNode};

    #[test]
    fn test_onnx_export_simple() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");
        let z = graph.add_tensor("Z");

        graph
            .add_node(EinsumNode::elem_binary("add", x, y, z))
            .unwrap();
        graph.add_output(z).unwrap();

        let onnx = export_to_onnx_text(&graph).unwrap();

        assert!(onnx.contains("ir_version"));
        assert!(onnx.contains("Add"));
        assert!(onnx.contains("X"));
        assert!(onnx.contains("Y"));
        assert!(onnx.contains("Z"));
    }

    #[test]
    fn test_onnx_export_einsum() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .unwrap();
        graph.add_output(c).unwrap();

        let onnx = export_to_onnx_text(&graph).unwrap();

        assert!(onnx.contains("Einsum"));
        assert!(onnx.contains("ij,jk->ik"));
    }

    #[test]
    fn test_torchscript_export_simple() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");
        let z = graph.add_tensor("Z");

        graph
            .add_node(EinsumNode::elem_binary("mul", x, y, z))
            .unwrap();
        graph.add_output(z).unwrap();

        let script = export_to_torchscript_text(&graph).unwrap();

        assert!(script.contains("import torch"));
        assert!(script.contains("class TensorLogicGraph"));
        assert!(script.contains("torch.mul"));
    }

    #[test]
    fn test_torchscript_export_einsum() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let w = graph.add_tensor("W");
        let y = graph.add_tensor("Y");

        graph
            .add_node(EinsumNode::einsum("ij,jk->ik", vec![x, w], vec![y]))
            .unwrap();
        graph.add_output(y).unwrap();

        let script = export_to_torchscript_text(&graph).unwrap();

        assert!(script.contains("torch.einsum"));
        assert!(script.contains("'ij,jk->ik'"));
    }

    #[test]
    fn test_onnx_export_reduction() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");

        graph
            .add_node(EinsumNode::reduce("sum", vec![0, 1], x, y))
            .unwrap();
        graph.add_output(y).unwrap();

        let onnx = export_to_onnx_text(&graph).unwrap();

        assert!(onnx.contains("ReduceSum"));
        assert!(onnx.contains("axes"));
    }

    #[test]
    fn test_torchscript_export_unary() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");

        graph
            .add_node(EinsumNode::elem_unary("relu", x, y))
            .unwrap();
        graph.add_output(y).unwrap();

        let script = export_to_torchscript_text(&graph).unwrap();

        assert!(script.contains("torch.relu"));
    }

    #[test]
    fn test_onnx_export_with_options() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");

        graph.add_node(EinsumNode::elem_unary("exp", x, y)).unwrap();
        graph.add_output(y).unwrap();

        let options = OnnxExportOptions {
            opset_version: 14,
            producer_name: "CustomProducer".to_string(),
            ..Default::default()
        };

        let onnx = export_to_onnx_text_with_options(&graph, &options).unwrap();

        assert!(onnx.contains("version: 14"));
        assert!(onnx.contains("CustomProducer"));
    }

    #[test]
    fn test_torchscript_export_without_comments() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");

        graph
            .add_node(EinsumNode::elem_unary("tanh", x, y))
            .unwrap();
        graph.add_output(y).unwrap();

        let options = TorchScriptExportOptions {
            include_comments: false,
            ..Default::default()
        };

        let script = export_to_torchscript_text_with_options(&graph, &options).unwrap();

        assert!(!script.contains("# "));
        assert!(script.contains("torch.tanh"));
    }

    #[test]
    fn test_export_multiple_outputs() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("X");
        let y = graph.add_tensor("Y");
        let z = graph.add_tensor("Z");

        graph.add_node(EinsumNode::elem_unary("exp", x, y)).unwrap();
        graph.add_node(EinsumNode::elem_unary("log", x, z)).unwrap();
        graph.add_output(y).unwrap();
        graph.add_output(z).unwrap();

        let script = export_to_torchscript_text(&graph).unwrap();

        assert!(script.contains("return (Y, Z)"));
    }
}
