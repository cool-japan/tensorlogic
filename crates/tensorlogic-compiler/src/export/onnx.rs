//! ONNX export functionality for TensorLogic computation graphs.
//!
//! This module provides functionality to export compiled `EinsumGraph` instances to
//! ONNX format, enabling execution on ONNX Runtime, PyTorch, TensorFlow, and other
//! ONNX-compatible frameworks.
//!
//! # Features
//!
//! - Maps einsum operations to ONNX Einsum operator
//! - Supports element-wise operations (Add, Sub, Mul, Div, etc.)
//! - Handles reduction operations (Sum, Max, Min)
//! - Automatic shape inference
//! - Proper tensor type definitions
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_compiler::export::onnx::{export_to_onnx, OnnxExportConfig};
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::and(
//!     TLExpr::pred("P", vec![Term::var("x")]),
//!     TLExpr::pred("Q", vec![Term::var("x")]),
//! );
//! let graph = compile_to_einsum(&expr)?;
//!
//! // Export with custom configuration
//! let config = OnnxExportConfig {
//!     model_name: "logic_model".to_string(),
//!     opset_version: 13,
//!     default_dtype: DataType::Float32,
//! };
//!
//! let onnx_bytes = export_to_onnx_with_config(&graph, config)?;
//! std::fs::write("model.onnx", onnx_bytes)?;
//! ```

use anyhow::{anyhow, Result};
use prost::Message;
use std::collections::HashMap;
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// ONNX data types supported for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// Boolean
    Bool,
}

impl DataType {
    /// Convert to ONNX TensorProto DataType enum value.
    fn to_onnx_enum(self) -> i32 {
        match self {
            DataType::Float32 => 1,  // FLOAT
            DataType::Float64 => 11, // DOUBLE
            DataType::Int32 => 6,    // INT32
            DataType::Int64 => 7,    // INT64
            DataType::Bool => 9,     // BOOL
        }
    }
}

/// Configuration for ONNX export.
#[derive(Debug, Clone)]
pub struct OnnxExportConfig {
    /// Name of the ONNX model
    pub model_name: String,
    /// ONNX opset version (default: 13)
    pub opset_version: i64,
    /// Default data type for tensors
    pub default_dtype: DataType,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            model_name: "tensorlogic_model".to_string(),
            opset_version: 13,
            default_dtype: DataType::Float32,
        }
    }
}

/// Minimal ONNX protobuf structures (simplified for our use case).
///
/// These definitions cover the essential ONNX protobuf messages needed for export.
/// Full ONNX spec: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
struct Dimension {
    #[prost(oneof = "DimensionValue", tags = "1, 2")]
    value: Option<DimensionValue>,
}

#[derive(Clone, PartialEq, ::prost::Oneof)]
enum DimensionValue {
    #[prost(int64, tag = "1")]
    DimValue(i64),
    #[prost(string, tag = "2")]
    DimParam(String),
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(oneof = "TypeProtoValue", tags = "1")]
    value: Option<TypeProtoValue>,
}

#[derive(Clone, PartialEq, ::prost::Oneof)]
enum TypeProtoValue {
    #[prost(message, tag = "1")]
    TensorType(TensorTypeProto),
}

#[derive(Clone, PartialEq, Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(int32, tag = "20")]
    r#type: i32,
    #[prost(string, tag = "2")]
    s: String,
    #[prost(int64, repeated, tag = "7")]
    ints: Vec<i64>,
    #[prost(float, repeated, tag = "6")]
    floats: Vec<f32>,
}

#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
    #[prost(string, tag = "10")]
    doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    producer_name: String,
    #[prost(string, tag = "3")]
    producer_version: String,
    #[prost(string, tag = "4")]
    domain: String,
    #[prost(int64, tag = "5")]
    model_version: i64,
    #[prost(string, tag = "6")]
    doc_string: String,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

/// Export an EinsumGraph to ONNX format with default configuration.
///
/// # Arguments
///
/// * `graph` - The compiled einsum graph to export
/// * `model_name` - Name for the ONNX model
///
/// # Returns
///
/// Serialized ONNX model as bytes, ready to be written to a .onnx file
///
/// # Example
///
/// ```rust,ignore
/// let graph = compile_to_einsum(&expr)?;
/// let onnx_bytes = export_to_onnx(&graph, "my_logic_model")?;
/// std::fs::write("model.onnx", onnx_bytes)?;
/// ```
pub fn export_to_onnx(graph: &EinsumGraph, model_name: &str) -> Result<Vec<u8>> {
    let config = OnnxExportConfig {
        model_name: model_name.to_string(),
        ..Default::default()
    };
    export_to_onnx_with_config(graph, config)
}

/// Export an EinsumGraph to ONNX format with custom configuration.
///
/// Provides fine-grained control over the export process including opset version,
/// data types, and model metadata.
///
/// # Arguments
///
/// * `graph` - The compiled einsum graph to export
/// * `config` - Export configuration
///
/// # Returns
///
/// Serialized ONNX model as bytes
pub fn export_to_onnx_with_config(
    graph: &EinsumGraph,
    config: OnnxExportConfig,
) -> Result<Vec<u8>> {
    let converter = OnnxConverter::new(config);
    converter.convert(graph)
}

/// Internal converter for EinsumGraph to ONNX.
struct OnnxConverter {
    config: OnnxExportConfig,
}

impl OnnxConverter {
    fn new(config: OnnxExportConfig) -> Self {
        Self { config }
    }

    fn convert(&self, graph: &EinsumGraph) -> Result<Vec<u8>> {
        let onnx_graph = self.build_graph(graph)?;

        let model = ModelProto {
            ir_version: 7, // ONNX IR version
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(), // Default domain
                version: self.config.opset_version,
            }],
            producer_name: "TensorLogic".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: "ai.tensorlogic".to_string(),
            model_version: 1,
            doc_string: format!("Compiled TensorLogic model: {}", self.config.model_name),
            graph: Some(onnx_graph),
        };

        let mut buf = Vec::new();
        model
            .encode(&mut buf)
            .map_err(|e| anyhow!("Failed to encode ONNX model: {}", e))?;

        Ok(buf)
    }

    fn build_graph(&self, graph: &EinsumGraph) -> Result<GraphProto> {
        let mut nodes = Vec::new();
        let mut tensor_to_onnx_name: HashMap<String, String> = HashMap::new();

        // Map input tensors
        for tensor in &graph.tensors {
            let onnx_name = self.sanitize_tensor_name(tensor);
            tensor_to_onnx_name.insert(tensor.clone(), onnx_name);
        }

        // Convert each EinsumNode to ONNX NodeProto
        for (idx, node) in graph.nodes.iter().enumerate() {
            let onnx_node = self.convert_node(node, idx, &tensor_to_onnx_name)?;
            nodes.push(onnx_node);
        }

        // Define graph inputs (predicates)
        let inputs: Vec<ValueInfoProto> = graph
            .tensors
            .iter()
            .enumerate()
            .filter(|(idx, _)| !graph.outputs.contains(idx))
            .map(|(_, tensor)| {
                let onnx_name = tensor_to_onnx_name.get(tensor).unwrap().clone();
                self.create_value_info(&onnx_name, &format!("Input tensor: {}", tensor))
            })
            .collect();

        // Define graph outputs
        let outputs: Vec<ValueInfoProto> = graph
            .outputs
            .iter()
            .map(|&idx| {
                let tensor = &graph.tensors[idx];
                let onnx_name = tensor_to_onnx_name.get(tensor).unwrap().clone();
                self.create_value_info(&onnx_name, &format!("Output tensor: {}", tensor))
            })
            .collect();

        Ok(GraphProto {
            node: nodes,
            name: self.config.model_name.clone(),
            input: inputs,
            output: outputs,
            doc_string: "TensorLogic compiled graph".to_string(),
        })
    }

    fn convert_node(
        &self,
        node: &EinsumNode,
        idx: usize,
        tensor_names: &HashMap<String, String>,
    ) -> Result<NodeProto> {
        use tensorlogic_ir::OpType;

        match &node.op {
            OpType::Einsum { spec } => {
                self.convert_einsum_node(spec, &node.inputs, &node.outputs, idx, tensor_names)
            }
            OpType::ElemUnary { op } => {
                self.convert_elem_unary_node(op, node.inputs[0], node.outputs[0], idx, tensor_names)
            }
            OpType::ElemBinary { op } => self.convert_elem_binary_node(
                op,
                node.inputs[0],
                node.inputs[1],
                node.outputs[0],
                idx,
                tensor_names,
            ),
            OpType::Reduce { op, axes } => self.convert_reduce_node(
                op,
                axes,
                node.inputs[0],
                node.outputs[0],
                idx,
                tensor_names,
            ),
        }
    }

    fn convert_einsum_node(
        &self,
        spec: &str,
        inputs: &[usize],
        outputs: &[usize],
        idx: usize,
        tensor_names: &HashMap<String, String>,
    ) -> Result<NodeProto> {
        let input_names: Vec<String> = inputs
            .iter()
            .map(|&i| tensor_names.values().nth(i).unwrap().clone())
            .collect();

        let output_name = if let Some(&out_idx) = outputs.first() {
            tensor_names.values().nth(out_idx).unwrap().clone()
        } else {
            format!("node_{}_out", idx)
        };

        let einsum_attr = AttributeProto {
            name: "equation".to_string(),
            r#type: 3, // STRING
            s: spec.to_string(),
            ints: vec![],
            floats: vec![],
        };

        Ok(NodeProto {
            input: input_names,
            output: vec![output_name],
            name: format!("einsum_{}", idx),
            op_type: "Einsum".to_string(),
            attribute: vec![einsum_attr],
            doc_string: format!("Einsum operation: {}", spec),
        })
    }

    fn convert_elem_unary_node(
        &self,
        op: &str,
        input: usize,
        output: usize,
        idx: usize,
        tensor_names: &HashMap<String, String>,
    ) -> Result<NodeProto> {
        let input_name = tensor_names.values().nth(input).unwrap().clone();
        let output_name = tensor_names.values().nth(output).unwrap().clone();

        let op_type = match op {
            "relu" => "Relu",
            "sigmoid" => "Sigmoid",
            "tanh" => "Tanh",
            "exp" => "Exp",
            "log" => "Log",
            "sqrt" => "Sqrt",
            "abs" => "Abs",
            "neg" => "Neg",
            "not" => "Not",
            _ => return Err(anyhow!("Unsupported unary operation: {}", op)),
        };

        Ok(NodeProto {
            input: vec![input_name],
            output: vec![output_name],
            name: format!("{}_{}", op, idx),
            op_type: op_type.to_string(),
            attribute: vec![],
            doc_string: format!("Unary operation: {}", op),
        })
    }

    fn convert_elem_binary_node(
        &self,
        op: &str,
        left: usize,
        right: usize,
        output: usize,
        idx: usize,
        tensor_names: &HashMap<String, String>,
    ) -> Result<NodeProto> {
        let left_name = tensor_names.values().nth(left).unwrap().clone();
        let right_name = tensor_names.values().nth(right).unwrap().clone();
        let output_name = tensor_names.values().nth(output).unwrap().clone();

        let op_type = match op {
            "add" => "Add",
            "subtract" => "Sub",
            "multiply" => "Mul",
            "divide" => "Div",
            "max" => "Max",
            "min" => "Min",
            "and" => "And",
            "or" => "Or",
            _ => return Err(anyhow!("Unsupported binary operation: {}", op)),
        };

        Ok(NodeProto {
            input: vec![left_name, right_name],
            output: vec![output_name],
            name: format!("{}_{}", op, idx),
            op_type: op_type.to_string(),
            attribute: vec![],
            doc_string: format!("Binary operation: {}", op),
        })
    }

    fn convert_reduce_node(
        &self,
        op: &str,
        axes: &[usize],
        input: usize,
        output: usize,
        idx: usize,
        tensor_names: &HashMap<String, String>,
    ) -> Result<NodeProto> {
        let input_name = tensor_names.values().nth(input).unwrap().clone();
        let output_name = tensor_names.values().nth(output).unwrap().clone();

        let op_type = match op {
            "sum" => "ReduceSum",
            "max" => "ReduceMax",
            "min" => "ReduceMin",
            "mean" => "ReduceMean",
            "prod" => "ReduceProd",
            _ => return Err(anyhow!("Unsupported reduce operation: {}", op)),
        };

        let axes_attr = AttributeProto {
            name: "axes".to_string(),
            r#type: 7, // INTS
            s: String::new(),
            ints: axes.iter().map(|&a| a as i64).collect(),
            floats: vec![],
        };

        Ok(NodeProto {
            input: vec![input_name],
            output: vec![output_name],
            name: format!("{}_{}", op, idx),
            op_type: op_type.to_string(),
            attribute: vec![axes_attr],
            doc_string: format!("Reduce operation: {}", op),
        })
    }

    fn create_value_info(&self, name: &str, doc: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(TypeProtoValue::TensorType(TensorTypeProto {
                    elem_type: self.config.default_dtype.to_onnx_enum(),
                    shape: None, // Dynamic shape
                })),
            }),
            doc_string: doc.to_string(),
        }
    }

    fn sanitize_tensor_name(&self, name: &str) -> String {
        // Replace invalid characters for ONNX names
        name.replace(['[', ']', ',', ' '], "_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode};

    #[test]
    fn test_data_type_conversion() {
        assert_eq!(DataType::Float32.to_onnx_enum(), 1);
        assert_eq!(DataType::Float64.to_onnx_enum(), 11);
        assert_eq!(DataType::Int32.to_onnx_enum(), 6);
        assert_eq!(DataType::Int64.to_onnx_enum(), 7);
        assert_eq!(DataType::Bool.to_onnx_enum(), 9);
    }

    #[test]
    fn test_onnx_export_config_default() {
        let config = OnnxExportConfig::default();
        assert_eq!(config.model_name, "tensorlogic_model");
        assert_eq!(config.opset_version, 13);
        assert_eq!(config.default_dtype, DataType::Float32);
    }

    #[test]
    fn test_export_simple_graph() {
        let mut graph = EinsumGraph::new();
        let p_idx = graph.add_tensor("P[a]");
        let q_idx = graph.add_tensor("Q[a]");
        let result_idx = graph.add_tensor("result");

        let _node = graph.add_node(EinsumNode::elem_binary(
            "multiply", p_idx, q_idx, result_idx,
        ));

        graph.outputs.push(result_idx);

        let result = export_to_onnx(&graph, "test_model");
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.len() > 10); // Should have reasonable size
    }

    #[test]
    fn test_export_einsum_operation() {
        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("A[ab]");
        let b_idx = graph.add_tensor("B[bc]");
        let result_idx = graph.add_tensor("result");

        let _node = graph.add_node(EinsumNode::einsum(
            "ab,bc->ac",
            vec![a_idx, b_idx],
            vec![result_idx],
        ));
        graph.outputs.push(result_idx);

        let result = export_to_onnx(&graph, "einsum_model");
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_with_custom_config() {
        let mut graph = EinsumGraph::new();
        let input_idx = graph.add_tensor("input");
        let output_idx = graph.add_tensor("output");
        let _node = graph.add_node(EinsumNode::elem_unary("relu", input_idx, output_idx));
        graph.outputs.push(output_idx);

        let config = OnnxExportConfig {
            model_name: "custom_model".to_string(),
            opset_version: 14,
            default_dtype: DataType::Float64,
        };

        let result = export_to_onnx_with_config(&graph, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sanitize_tensor_name() {
        let converter = OnnxConverter::new(OnnxExportConfig::default());
        assert_eq!(converter.sanitize_tensor_name("P[a,b]"), "P_a_b_");
        assert_eq!(converter.sanitize_tensor_name("result"), "result");
        assert_eq!(converter.sanitize_tensor_name("temp[0]"), "temp_0_");
    }

    #[test]
    fn test_export_reduce_node() {
        let mut graph = EinsumGraph::new();
        let input_idx = graph.add_tensor("input[ab]");
        let output_idx = graph.add_tensor("output");
        let _node = graph.add_node(EinsumNode::reduce("sum", vec![1], input_idx, output_idx));
        graph.outputs.push(output_idx);

        let result = export_to_onnx(&graph, "reduce_model");
        assert!(result.is_ok());
    }
}
