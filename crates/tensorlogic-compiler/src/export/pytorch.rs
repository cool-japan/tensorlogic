//! PyTorch code generation for TensorLogic computation graphs.
//!
//! This module provides functionality to generate PyTorch `nn.Module` Python code from
//! compiled `EinsumGraph` instances. The generated code can be used directly in PyTorch
//! workflows, traced with `torch.jit.trace`, or scripted with `torch.jit.script`.
//!
//! # Features
//!
//! - Generates complete PyTorch `nn.Module` classes from EinsumGraph
//! - Maps einsum operations to `torch.einsum`
//! - Supports element-wise operations (add, sub, mul, div, etc.)
//! - Handles reduction operations (sum, max, min, mean)
//! - Generates human-readable, editable Python code
//! - Supports both eager execution and TorchScript compilation
//! - Configurable tensor dtypes (float32, float64, int32, int64, bool)
//!
//! # PyTorch Integration
//!
//! The generated code can be:
//! - Executed directly in PyTorch eager mode
//! - Traced with `torch.jit.trace()` for optimization
//! - Scripted with `torch.jit.script()` for deployment
//! - Exported to ONNX via PyTorch's export functionality
//! - Used in training loops or inference pipelines
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_compiler::export::pytorch::export_to_pytorch;
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::and(
//!     TLExpr::pred("P", vec![Term::var("x")]),
//!     TLExpr::pred("Q", vec![Term::var("x")]),
//! );
//! let graph = compile_to_einsum(&expr)?;
//!
//! // Generate PyTorch code
//! let pytorch_code = export_to_pytorch(&graph, "LogicModel")?;
//! std::fs::write("model.py", pytorch_code)?;
//! ```
//!
//! The generated Python code can then be used:
//!
//! ```python
//! import torch
//! from model import LogicModel
//!
//! model = LogicModel()
//! inputs = {"P": torch.rand(10), "Q": torch.rand(10)}
//! output = model(inputs)
//!
//! # Trace for TorchScript
//! traced = torch.jit.trace(model, inputs)
//! traced.save("model.pt")
//! ```

use anyhow::{anyhow, Result};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// PyTorch tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTorchDtype {
    /// 32-bit floating point (torch.float32)
    Float32,
    /// 64-bit floating point (torch.float64)
    Float64,
    /// 32-bit integer (torch.int32)
    Int32,
    /// 64-bit integer (torch.int64)
    Int64,
    /// Boolean (torch.bool)
    Bool,
}

impl PyTorchDtype {
    /// Get the PyTorch dtype string.
    fn to_torch_string(self) -> &'static str {
        match self {
            PyTorchDtype::Float32 => "torch.float32",
            PyTorchDtype::Float64 => "torch.float64",
            PyTorchDtype::Int32 => "torch.int32",
            PyTorchDtype::Int64 => "torch.int64",
            PyTorchDtype::Bool => "torch.bool",
        }
    }
}

/// Configuration for PyTorch code generation.
#[derive(Debug, Clone)]
pub struct PyTorchExportConfig {
    /// Name of the generated PyTorch module class
    pub class_name: String,
    /// Default data type for tensors
    pub default_dtype: PyTorchDtype,
    /// Whether to add TorchScript decorators (@torch.jit.script)
    pub add_jit_decorators: bool,
    /// Indentation string (default: 4 spaces)
    pub indent: String,
}

impl Default for PyTorchExportConfig {
    fn default() -> Self {
        Self {
            class_name: "TensorLogicModel".to_string(),
            default_dtype: PyTorchDtype::Float32,
            add_jit_decorators: false,
            indent: "    ".to_string(),
        }
    }
}

/// Code generator for translating EinsumGraph to PyTorch Python code.
struct PyTorchCodeGen {
    config: PyTorchExportConfig,
    code: Vec<String>,
    indent_level: usize,
}

impl PyTorchCodeGen {
    fn new(config: PyTorchExportConfig) -> Self {
        Self {
            config,
            code: Vec::new(),
            indent_level: 0,
        }
    }

    fn indent(&self) -> String {
        self.config.indent.repeat(self.indent_level)
    }

    fn writeln(&mut self, line: impl AsRef<str>) {
        let line = line.as_ref();
        if line.is_empty() {
            self.code.push(String::new());
        } else {
            self.code.push(format!("{}{}", self.indent(), line));
        }
    }

    fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    fn decrease_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    fn generate_imports(&mut self) {
        self.writeln("\"\"\"");
        self.writeln("Auto-generated PyTorch model from TensorLogic compilation.");
        self.writeln("");
        self.writeln("This module defines a PyTorch nn.Module that implements the logic");
        self.writeln("expressed in the original TensorLogic expression.");
        self.writeln("\"\"\"");
        self.writeln("");
        self.writeln("import torch");
        self.writeln("import torch.nn as nn");
        self.writeln("from typing import Dict, Tuple");
        self.writeln("");
    }

    fn generate_class_header(&mut self) {
        self.writeln("");
        self.writeln(format!("class {}(nn.Module):", self.config.class_name));
        self.increase_indent();

        self.writeln("\"\"\"");
        self.writeln("TensorLogic computation graph compiled to PyTorch.");
        self.writeln("");
        self.writeln("This model can be used in eager mode or compiled with TorchScript.");
        self.writeln("\"\"\"");
        self.writeln("");
    }

    fn generate_init(&mut self, graph: &EinsumGraph) {
        self.writeln("def __init__(self):");
        self.increase_indent();
        self.writeln("super().__init__()");
        self.writeln(format!(
            "self.dtype = {}",
            self.config.default_dtype.to_torch_string()
        ));

        // Collect input tensor names
        let input_names: Vec<String> = graph
            .inputs
            .iter()
            .filter_map(|&idx| graph.tensors.get(idx).cloned())
            .collect();

        if !input_names.is_empty() {
            self.writeln(format!("self.input_names = {:?}", input_names));
        }

        self.decrease_indent();
        self.writeln("");
    }

    fn generate_forward(&mut self, graph: &EinsumGraph) -> Result<()> {
        if self.config.add_jit_decorators {
            self.writeln("@torch.jit.export");
        }

        self.writeln("def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:");
        self.increase_indent();
        self.writeln("\"\"\"");
        self.writeln("Forward pass through the logic computation graph.");
        self.writeln("");
        self.writeln("Args:");
        self.writeln("    inputs: Dictionary mapping tensor names to torch.Tensor values");
        self.writeln("");
        self.writeln("Returns:");
        self.writeln("    Output tensor from the computation");
        self.writeln("\"\"\"");
        self.writeln("");

        // Find which tensors are produced by nodes (not inputs)
        let mut produced_tensors = std::collections::HashSet::new();
        for node in &graph.nodes {
            for &output_idx in &node.outputs {
                produced_tensors.insert(output_idx);
            }
        }

        // Generate input variable assignments for tensors not produced by any node
        let mut has_inputs = false;
        for (idx, name) in graph.tensors.iter().enumerate() {
            if !produced_tensors.contains(&idx) && !graph.outputs.contains(&idx) {
                let safe_name = make_safe_identifier(name);
                self.writeln(format!(
                    "{} = inputs.get('{}', inputs.get('{}'))",
                    safe_name, name, safe_name
                ));
                has_inputs = true;
            }
        }

        if has_inputs {
            self.writeln("");
        }

        // Generate intermediate tensor computations
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.generate_node_computation(node, idx, &graph.tensors, &produced_tensors)?;
        }

        // Return the output tensor(s)
        if !graph.outputs.is_empty() {
            let output_name = &graph.tensors[graph.outputs[0]];
            self.writeln(format!("return {}", make_safe_identifier(output_name)));
        } else {
            self.writeln("return None  # No outputs specified");
        }

        self.decrease_indent();
        self.writeln("");

        Ok(())
    }

    fn generate_node_computation(
        &mut self,
        node: &EinsumNode,
        _idx: usize,
        tensor_names: &[String],
        _produced_tensors: &std::collections::HashSet<usize>,
    ) -> Result<()> {
        match &node.op {
            OpType::Einsum { spec } => {
                if !node.outputs.is_empty() {
                    let output_name = make_safe_identifier(&tensor_names[node.outputs[0]]);
                    let input_refs: Vec<String> = node
                        .inputs
                        .iter()
                        .map(|&i| make_safe_identifier(&tensor_names[i]))
                        .collect();

                    if input_refs.is_empty() {
                        return Ok(());
                    }

                    self.writeln(format!(
                        "{} = torch.einsum('{}', {})",
                        output_name,
                        spec,
                        input_refs.join(", ")
                    ));
                }
            }
            OpType::ElemUnary { op } => {
                if !node.inputs.is_empty() && !node.outputs.is_empty() {
                    let input_name = make_safe_identifier(&tensor_names[node.inputs[0]]);
                    let output_name = make_safe_identifier(&tensor_names[node.outputs[0]]);

                    self.generate_unary_op(op, &input_name, &output_name)?;
                }
            }
            OpType::ElemBinary { op } => {
                if node.inputs.len() >= 2 && !node.outputs.is_empty() {
                    let left_name = make_safe_identifier(&tensor_names[node.inputs[0]]);
                    let right_name = make_safe_identifier(&tensor_names[node.inputs[1]]);
                    let output_name = make_safe_identifier(&tensor_names[node.outputs[0]]);

                    self.generate_binary_op(op, &left_name, &right_name, &output_name)?;
                }
            }
            OpType::Reduce { op, axes } => {
                if !node.inputs.is_empty() && !node.outputs.is_empty() {
                    let input_name = make_safe_identifier(&tensor_names[node.inputs[0]]);
                    let output_name = make_safe_identifier(&tensor_names[node.outputs[0]]);

                    self.generate_reduce_op(op, axes, &input_name, &output_name)?;
                }
            }
        }

        Ok(())
    }

    fn generate_unary_op(&mut self, op: &str, input: &str, output: &str) -> Result<()> {
        // Handle special case: one_minus
        if op == "one_minus" || op == "oneminus" {
            self.writeln(format!("{} = 1.0 - {}", output, input));
            return Ok(());
        }

        let torch_op = match op {
            "exp" => "torch.exp",
            "log" => "torch.log",
            "sqrt" => "torch.sqrt",
            "abs" => "torch.abs",
            "neg" | "negate" => "torch.neg",
            "sigmoid" => "torch.sigmoid",
            "tanh" => "torch.tanh",
            "sin" => "torch.sin",
            "cos" => "torch.cos",
            "tan" => "torch.tan",
            "floor" => "torch.floor",
            "ceil" => "torch.ceil",
            "round" => "torch.round",
            "relu" => "torch.nn.functional.relu",
            "not" => "torch.logical_not",
            _ => return Err(anyhow!("Unsupported unary operation for PyTorch: {}", op)),
        };

        self.writeln(format!("{} = {}({})", output, torch_op, input));
        Ok(())
    }

    fn generate_binary_op(
        &mut self,
        op: &str,
        left: &str,
        right: &str,
        output: &str,
    ) -> Result<()> {
        let torch_expr = match op {
            "add" => format!("{} + {}", left, right),
            "sub" | "subtract" => format!("{} - {}", left, right),
            "mul" | "multiply" => format!("{} * {}", left, right),
            "div" | "divide" => format!("{} / {}", left, right),
            "pow" | "power" => format!("torch.pow({}, {})", left, right),
            "max" | "maximum" => format!("torch.maximum({}, {})", left, right),
            "min" | "minimum" => format!("torch.minimum({}, {})", left, right),
            "eq" | "equal" => format!("torch.eq({}, {})", left, right),
            "lt" | "less" => format!("torch.lt({}, {})", left, right),
            "gt" | "greater" => format!("torch.gt({}, {})", left, right),
            "lte" | "less_equal" => format!("torch.le({}, {})", left, right),
            "gte" | "greater_equal" => format!("torch.ge({}, {})", left, right),
            "and" => format!("torch.logical_and({}, {})", left, right),
            "or" => format!("torch.logical_or({}, {})", left, right),
            _ => return Err(anyhow!("Unsupported binary operation for PyTorch: {}", op)),
        };

        self.writeln(format!("{} = {}", output, torch_expr));
        Ok(())
    }

    fn generate_reduce_op(
        &mut self,
        op: &str,
        axes: &[usize],
        input: &str,
        output: &str,
    ) -> Result<()> {
        let axes_list = if axes.is_empty() {
            "None".to_string()
        } else if axes.len() == 1 {
            axes[0].to_string()
        } else {
            format!(
                "[{}]",
                axes.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let torch_expr = match op {
            "sum" => format!("torch.sum({}, dim={})", input, axes_list),
            "max" => format!("torch.max({}, dim={})[0]", input, axes_list),
            "min" => format!("torch.min({}, dim={})[0]", input, axes_list),
            "mean" => format!("torch.mean({}, dim={})", input, axes_list),
            "prod" | "product" => format!("torch.prod({}, dim={})", input, axes_list),
            "any" => format!("torch.any({}, dim={})", input, axes_list),
            "all" => format!("torch.all({}, dim={})", input, axes_list),
            _ => return Err(anyhow!("Unsupported reduce operation for PyTorch: {}", op)),
        };

        self.writeln(format!("{} = {}", output, torch_expr));
        Ok(())
    }

    fn generate(&mut self, graph: &EinsumGraph) -> Result<String> {
        self.generate_imports();
        self.generate_class_header();
        self.generate_init(graph);
        self.generate_forward(graph)?;

        // Close class definition
        self.decrease_indent();

        // Add convenience function
        self.writeln("");
        self.writeln("");
        self.writeln("def create_model():");
        self.increase_indent();
        self.writeln(format!(
            "\"\"\"Create a new {} instance.\"\"\"",
            self.config.class_name
        ));
        self.writeln(format!("return {}()", self.config.class_name));
        self.decrease_indent();

        Ok(self.code.join("\n"))
    }
}

/// Make a safe Python identifier from a tensor name.
fn make_safe_identifier(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    // Ensure it doesn't start with a digit
    if sanitized.chars().next().is_some_and(|c| c.is_numeric()) {
        format!("t_{}", sanitized)
    } else {
        sanitized
    }
}

/// Sanitize tensor names to be valid Python identifiers (test helper).
#[cfg(test)]
fn sanitize_name(name: &str) -> String {
    make_safe_identifier(name)
}

/// Export an EinsumGraph to PyTorch Python code.
///
/// Generates a complete PyTorch nn.Module class that implements the computation graph.
///
/// # Example
///
/// ```rust,ignore
/// use tensorlogic_compiler::export::pytorch::export_to_pytorch;
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let graph = compile_to_einsum(&expr)?;
/// let pytorch_code = export_to_pytorch(&graph, "KnowledgeModel")?;
/// std::fs::write("model.py", pytorch_code)?;
/// ```
pub fn export_to_pytorch(graph: &EinsumGraph, class_name: &str) -> Result<String> {
    let config = PyTorchExportConfig {
        class_name: class_name.to_string(),
        ..Default::default()
    };
    export_to_pytorch_with_config(graph, config)
}

/// Export an EinsumGraph to PyTorch Python code with custom configuration.
///
/// Allows fine-grained control over code generation (class name, dtype, indentation, etc.).
///
/// # Example
///
/// ```rust,ignore
/// use tensorlogic_compiler::export::pytorch::{
///     export_to_pytorch_with_config, PyTorchExportConfig, PyTorchDtype
/// };
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let graph = compile_to_einsum(&expr)?;
///
/// let config = PyTorchExportConfig {
///     class_name: "LogicModel".to_string(),
///     default_dtype: PyTorchDtype::Float64,
///     add_jit_decorators: true,
///     indent: "  ".to_string(),  // 2-space indentation
/// };
///
/// let pytorch_code = export_to_pytorch_with_config(&graph, config)?;
/// ```
pub fn export_to_pytorch_with_config(
    graph: &EinsumGraph,
    config: PyTorchExportConfig,
) -> Result<String> {
    let mut codegen = PyTorchCodeGen::new(config);
    codegen.generate(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode};

    #[test]
    fn test_pytorch_dtype_strings() {
        assert_eq!(PyTorchDtype::Float32.to_torch_string(), "torch.float32");
        assert_eq!(PyTorchDtype::Float64.to_torch_string(), "torch.float64");
        assert_eq!(PyTorchDtype::Int32.to_torch_string(), "torch.int32");
        assert_eq!(PyTorchDtype::Int64.to_torch_string(), "torch.int64");
        assert_eq!(PyTorchDtype::Bool.to_torch_string(), "torch.bool");
    }

    #[test]
    fn test_default_config() {
        let config = PyTorchExportConfig::default();
        assert_eq!(config.class_name, "TensorLogicModel");
        assert_eq!(config.default_dtype, PyTorchDtype::Float32);
        assert!(!config.add_jit_decorators);
        assert_eq!(config.indent, "    ");
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("valid_name"), "valid_name");
        assert_eq!(sanitize_name("tensor_0"), "tensor_0");
        assert_eq!(sanitize_name("temp_1"), "temp_1");
        assert_eq!(sanitize_name("123invalid"), "t_123invalid");
        assert_eq!(sanitize_name("Pred-with-dash"), "Pred_with_dash");
    }

    #[test]
    fn test_export_simple_einsum() {
        let mut graph = EinsumGraph::new();

        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        let _node = graph
            .add_node(EinsumNode::einsum("ab,bc->ac", vec![a, b], vec![c]))
            .unwrap();
        graph.outputs.push(c);

        let result = export_to_pytorch(&graph, "SimpleModel");
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("class SimpleModel(nn.Module):"));
        assert!(code.contains("torch.einsum"));
        assert!(code.contains("ab,bc->ac"));
    }

    #[test]
    fn test_export_elem_binary() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");
        let t3 = graph.add_tensor("tensor_3");

        let _node = graph
            .add_node(EinsumNode::elem_binary("add", t1, t2, t3))
            .unwrap();
        graph.outputs.push(t3);

        let result = export_to_pytorch(&graph, "AddModel");
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("tensor_1 + tensor_2"));
    }

    #[test]
    fn test_export_elem_unary() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");

        let _node = graph
            .add_node(EinsumNode::elem_unary("exp", t1, t2))
            .unwrap();
        graph.outputs.push(t2);

        let result = export_to_pytorch(&graph, "ExpModel");
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("torch.exp"));
    }

    #[test]
    fn test_export_reduce() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");

        let _node = graph
            .add_node(EinsumNode::reduce("sum", vec![1], t1, t2))
            .unwrap();
        graph.outputs.push(t2);

        let result = export_to_pytorch(&graph, "SumModel");
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("torch.sum"));
        assert!(code.contains("dim=1"));
    }

    #[test]
    fn test_export_with_custom_config() {
        let mut graph = EinsumGraph::new();
        let t1 = graph.add_tensor("tensor_1");
        graph.outputs.push(t1);

        let config = PyTorchExportConfig {
            class_name: "CustomModel".to_string(),
            default_dtype: PyTorchDtype::Float64,
            add_jit_decorators: true,
            indent: "  ".to_string(),
        };

        let result = export_to_pytorch_with_config(&graph, config);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("class CustomModel(nn.Module):"));
        assert!(code.contains("torch.float64"));
        assert!(code.contains("@torch.jit.export"));
    }

    #[test]
    fn test_unsupported_unary_op() {
        let mut graph = EinsumGraph::new();
        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");

        let _node = graph
            .add_node(EinsumNode::elem_unary("invalid_op", t1, t2))
            .unwrap();
        graph.outputs.push(t2);

        let result = export_to_pytorch(&graph, "InvalidModel");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_binary_op() {
        let mut graph = EinsumGraph::new();
        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");
        let t3 = graph.add_tensor("tensor_3");

        let _node = graph
            .add_node(EinsumNode::elem_binary("invalid_op", t1, t2, t3))
            .unwrap();
        graph.outputs.push(t3);

        let result = export_to_pytorch(&graph, "InvalidModel");
        assert!(result.is_err());
    }

    #[test]
    fn test_one_minus_operation() {
        let mut graph = EinsumGraph::new();
        let t1 = graph.add_tensor("tensor_1");
        let t2 = graph.add_tensor("tensor_2");

        let _node = graph
            .add_node(EinsumNode::elem_unary("one_minus", t1, t2))
            .unwrap();
        graph.outputs.push(t2);

        let result = export_to_pytorch(&graph, "OneMinusModel");
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("1.0 - "));
    }
}
