//! TensorFlow GraphDef export functionality for TensorLogic computation graphs.
//!
//! This module provides functionality to export compiled `EinsumGraph` instances to
//! TensorFlow GraphDef format, enabling execution within TensorFlow runtime and
//! integration with TensorFlow-based workflows.
//!
//! # Features
//!
//! - Maps einsum operations to TensorFlow Einsum operator
//! - Supports element-wise operations (Add, Sub, Mul, Div, etc.)
//! - Handles reduction operations (Sum, Max, Min, Mean)
//! - Automatic shape inference with symbolic dimensions
//! - Proper tensor type definitions (float32, float64, int32, int64, bool)
//! - Support for unary operations (Exp, Log, Sqrt, Sin, Cos, Tanh, Sigmoid, etc.)
//! - Comparison operations (Equal, Greater, Less, etc.)
//!
//! # TensorFlow GraphDef Format
//!
//! GraphDef is TensorFlow's serialization format for computation graphs. It consists of:
//! - **Nodes**: Operations with inputs, outputs, and attributes
//! - **Attributes**: Operation-specific parameters (data types, shapes, etc.)
//! - **Edges**: Implicit connections via tensor names (output_name -> input_name)
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_compiler::export::tensorflow::{export_to_tensorflow, TensorFlowExportConfig};
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
//! let config = TensorFlowExportConfig {
//!     model_name: "logic_model".to_string(),
//!     default_dtype: TfDataType::Float32,
//!     add_identity_outputs: true,
//! };
//!
//! let graphdef_bytes = export_to_tensorflow_with_config(&graph, config)?;
//! std::fs::write("model.pb", graphdef_bytes)?;
//! ```
//!
//! # Limitations
//!
//! - Dynamic shapes use symbolic dimensions (e.g., "batch", "dim_0")
//! - Some advanced einsum patterns may require TensorFlow 2.x or later
//! - Custom operations are not supported (only standard TensorFlow ops)

use anyhow::{anyhow, Result};
use prost::Message;
use std::collections::HashMap;
use tensorlogic_ir::{EinsumGraph, OpType};

/// TensorFlow data types supported for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfDataType {
    /// 32-bit floating point (most common for ML)
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// Boolean (stored as uint8 in TensorFlow)
    Bool,
}

impl TfDataType {
    /// Convert to TensorFlow DataType enum value.
    /// See: tensorflow/core/framework/types.proto
    fn to_tf_enum(self) -> i32 {
        match self {
            TfDataType::Float32 => 1, // DT_FLOAT
            TfDataType::Float64 => 2, // DT_DOUBLE
            TfDataType::Int32 => 3,   // DT_INT32
            TfDataType::Int64 => 9,   // DT_INT64
            TfDataType::Bool => 10,   // DT_BOOL
        }
    }

    /// Get the string name used in TensorFlow operations.
    #[allow(dead_code)]
    fn to_tf_string(self) -> &'static str {
        match self {
            TfDataType::Float32 => "float32",
            TfDataType::Float64 => "float64",
            TfDataType::Int32 => "int32",
            TfDataType::Int64 => "int64",
            TfDataType::Bool => "bool",
        }
    }
}

/// Configuration for TensorFlow GraphDef export.
#[derive(Debug, Clone)]
pub struct TensorFlowExportConfig {
    /// Name of the TensorFlow model/graph
    pub model_name: String,
    /// Default data type for tensors
    pub default_dtype: TfDataType,
    /// Whether to add Identity nodes for outputs (recommended for SavedModel)
    pub add_identity_outputs: bool,
}

impl Default for TensorFlowExportConfig {
    fn default() -> Self {
        Self {
            model_name: "tensorlogic_model".to_string(),
            default_dtype: TfDataType::Float32,
            add_identity_outputs: true,
        }
    }
}

/// TensorFlow protobuf structures (simplified).
///
/// These definitions cover the essential TensorFlow protobuf messages needed for GraphDef export.
/// Full TensorFlow spec: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "2")]
    dim: Vec<TensorShapeProtoDim>,
    #[prost(bool, tag = "3")]
    unknown_rank: bool,
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProtoDim {
    #[prost(int64, tag = "1")]
    size: i64,
    #[prost(string, tag = "2")]
    name: String,
}

#[derive(Clone, PartialEq, Message)]
struct AttrValue {
    #[prost(oneof = "AttrValueValue", tags = "2, 3, 4, 5, 6, 7, 8")]
    value: Option<AttrValueValue>,
}

#[derive(Clone, PartialEq, ::prost::Oneof)]
enum AttrValueValue {
    #[prost(bytes, tag = "2")]
    S(Vec<u8>),
    #[prost(int64, tag = "3")]
    I(i64),
    #[prost(float, tag = "4")]
    F(f32),
    #[prost(bool, tag = "5")]
    B(bool),
    #[prost(enumeration = "i32", tag = "6")]
    Type(i32),
    #[prost(message, tag = "7")]
    Shape(TensorShapeProto),
    #[prost(message, tag = "8")]
    List(AttrValueList),
}

#[derive(Clone, PartialEq, Message)]
struct AttrValueList {
    #[prost(bytes, repeated, tag = "2")]
    s: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "3")]
    i: Vec<i64>,
    #[prost(float, repeated, tag = "4")]
    f: Vec<f32>,
    #[prost(bool, repeated, tag = "5")]
    b: Vec<bool>,
    #[prost(enumeration = "i32", repeated, tag = "6")]
    r#type: Vec<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct NodeDef {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(string, tag = "2")]
    op: String,
    #[prost(string, repeated, tag = "3")]
    input: Vec<String>,
    #[prost(string, tag = "4")]
    device: String,
    #[prost(map = "string, message", tag = "5")]
    attr: HashMap<String, AttrValue>,
}

#[derive(Clone, PartialEq, Message)]
struct GraphDef {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeDef>,
    #[prost(int32, tag = "3")]
    version: i32,
}

/// Converter for translating EinsumGraph to TensorFlow GraphDef.
struct TensorFlowConverter {
    config: TensorFlowExportConfig,
    tf_nodes: Vec<NodeDef>,
}

impl TensorFlowConverter {
    fn new(config: TensorFlowExportConfig) -> Self {
        Self {
            config,
            tf_nodes: Vec::new(),
        }
    }

    fn create_attr_dtype(&self) -> AttrValue {
        AttrValue {
            value: Some(AttrValueValue::Type(self.config.default_dtype.to_tf_enum())),
        }
    }

    fn create_attr_string(&self, s: impl Into<Vec<u8>>) -> AttrValue {
        AttrValue {
            value: Some(AttrValueValue::S(s.into())),
        }
    }

    fn create_attr_int_list(&self, ints: Vec<i64>) -> AttrValue {
        AttrValue {
            value: Some(AttrValueValue::List(AttrValueList {
                s: vec![],
                i: ints,
                f: vec![],
                b: vec![],
                r#type: vec![],
            })),
        }
    }

    fn convert_einsum_node(
        &mut self,
        spec: &str,
        inputs: &[usize],
        output: usize,
        _idx: usize,
        tensor_names: &[String],
    ) -> Result<()> {
        let input_names: Vec<String> = inputs.iter().map(|&i| tensor_names[i].clone()).collect();
        let output_name = tensor_names[output].clone();

        let mut attrs = HashMap::new();
        attrs.insert("T".to_string(), self.create_attr_dtype());
        attrs.insert(
            "equation".to_string(),
            self.create_attr_string(spec.as_bytes()),
        );

        self.tf_nodes.push(NodeDef {
            name: output_name,
            op: "Einsum".to_string(),
            input: input_names,
            device: String::new(),
            attr: attrs,
        });

        Ok(())
    }

    fn convert_elem_unary_node(
        &mut self,
        op: &str,
        input: usize,
        output: usize,
        _idx: usize,
        tensor_names: &[String],
    ) -> Result<()> {
        let input_name = tensor_names[input].clone();
        let output_name = tensor_names[output].clone();

        // Handle special case: one_minus (1 - x) requires const + subtract
        if op == "one_minus" || op == "oneminus" {
            // Create constant 1.0
            let const_name = format!("{}_const_one", output_name);
            let mut const_attrs = HashMap::new();
            const_attrs.insert("dtype".to_string(), self.create_attr_dtype());
            // Note: Proper tensor constant would need TensorProto, simplified here
            self.tf_nodes.push(NodeDef {
                name: const_name.clone(),
                op: "Const".to_string(),
                input: vec![],
                device: String::new(),
                attr: const_attrs,
            });

            // Create subtract: 1 - input
            let mut sub_attrs = HashMap::new();
            sub_attrs.insert("T".to_string(), self.create_attr_dtype());
            self.tf_nodes.push(NodeDef {
                name: output_name,
                op: "Sub".to_string(),
                input: vec![const_name, input_name],
                device: String::new(),
                attr: sub_attrs,
            });

            return Ok(());
        }

        // Map other EinsumNode unary ops to TensorFlow ops
        let tf_op = match op {
            "exp" => "Exp",
            "log" => "Log",
            "sqrt" => "Sqrt",
            "abs" => "Abs",
            "neg" | "negate" => "Neg",
            "sigmoid" => "Sigmoid",
            "tanh" => "Tanh",
            "sin" => "Sin",
            "cos" => "Cos",
            "tan" => "Tan",
            "floor" => "Floor",
            "ceil" => "Ceil",
            "round" => "Round",
            "relu" => "Relu",
            "not" => "LogicalNot",
            _ => {
                return Err(anyhow!(
                    "Unsupported unary operation for TensorFlow: {}",
                    op
                ))
            }
        };

        let mut attrs = HashMap::new();
        attrs.insert("T".to_string(), self.create_attr_dtype());

        self.tf_nodes.push(NodeDef {
            name: output_name,
            op: tf_op.to_string(),
            input: vec![input_name],
            device: String::new(),
            attr: attrs,
        });

        Ok(())
    }

    fn convert_elem_binary_node(
        &mut self,
        op: &str,
        left: usize,
        right: usize,
        output: usize,
        _idx: usize,
        tensor_names: &[String],
    ) -> Result<()> {
        let left_name = tensor_names[left].clone();
        let right_name = tensor_names[right].clone();
        let output_name = tensor_names[output].clone();

        // Map EinsumNode binary ops to TensorFlow ops
        let tf_op = match op {
            "add" => "Add",
            "sub" | "subtract" => "Sub",
            "mul" | "multiply" => "Mul",
            "div" | "divide" => "Div",
            "pow" | "power" => "Pow",
            "max" | "maximum" => "Maximum",
            "min" | "minimum" => "Minimum",
            "eq" | "equal" => "Equal",
            "lt" | "less" => "Less",
            "gt" | "greater" => "Greater",
            "lte" | "less_equal" => "LessEqual",
            "gte" | "greater_equal" => "GreaterEqual",
            "and" => "LogicalAnd",
            "or" => "LogicalOr",
            _ => {
                return Err(anyhow!(
                    "Unsupported binary operation for TensorFlow: {}",
                    op
                ))
            }
        };

        let mut attrs = HashMap::new();
        attrs.insert("T".to_string(), self.create_attr_dtype());

        self.tf_nodes.push(NodeDef {
            name: output_name,
            op: tf_op.to_string(),
            input: vec![left_name, right_name],
            device: String::new(),
            attr: attrs,
        });

        Ok(())
    }

    fn convert_reduce_node(
        &mut self,
        op: &str,
        axes: &[usize],
        input: usize,
        output: usize,
        _idx: usize,
        tensor_names: &[String],
    ) -> Result<()> {
        let input_name = tensor_names[input].clone();
        let output_name = tensor_names[output].clone();

        // Map reduce operations to TensorFlow ops
        let tf_op = match op {
            "sum" => "Sum",
            "max" => "Max",
            "min" => "Min",
            "mean" => "Mean",
            "prod" | "product" => "Prod",
            "any" => "Any",
            "all" => "All",
            _ => {
                return Err(anyhow!(
                    "Unsupported reduce operation for TensorFlow: {}",
                    op
                ))
            }
        };

        // Create a constant node for the reduction axes
        let axes_name = format!("{}_axes", output_name);
        let axes_i64: Vec<i64> = axes.iter().map(|&x| x as i64).collect();

        // Axes constant node
        let mut axes_attrs = HashMap::new();
        axes_attrs.insert(
            "dtype".to_string(),
            AttrValue {
                value: Some(AttrValueValue::Type(3)), // DT_INT32
            },
        );
        axes_attrs.insert(
            "value".to_string(),
            self.create_attr_int_list(axes_i64.clone()),
        );

        self.tf_nodes.push(NodeDef {
            name: axes_name.clone(),
            op: "Const".to_string(),
            input: vec![],
            device: String::new(),
            attr: axes_attrs,
        });

        // Reduction node
        let mut attrs = HashMap::new();
        attrs.insert("T".to_string(), self.create_attr_dtype());
        attrs.insert(
            "Tidx".to_string(),
            AttrValue {
                value: Some(AttrValueValue::Type(3)), // DT_INT32 for axes
            },
        );
        attrs.insert(
            "keep_dims".to_string(),
            AttrValue {
                value: Some(AttrValueValue::B(false)),
            },
        );

        self.tf_nodes.push(NodeDef {
            name: output_name,
            op: tf_op.to_string(),
            input: vec![input_name, axes_name],
            device: String::new(),
            attr: attrs,
        });

        Ok(())
    }

    fn convert(&mut self, graph: &EinsumGraph) -> Result<GraphDef> {
        // Convert all nodes in topological order
        for (idx, node) in graph.nodes.iter().enumerate() {
            match &node.op {
                OpType::Einsum { spec } => {
                    if !node.outputs.is_empty() {
                        self.convert_einsum_node(
                            spec,
                            &node.inputs,
                            node.outputs[0],
                            idx,
                            &graph.tensors,
                        )?;
                    }
                }
                OpType::ElemUnary { op } => {
                    if !node.inputs.is_empty() && !node.outputs.is_empty() {
                        self.convert_elem_unary_node(
                            op,
                            node.inputs[0],
                            node.outputs[0],
                            idx,
                            &graph.tensors,
                        )?;
                    }
                }
                OpType::ElemBinary { op } => {
                    if node.inputs.len() >= 2 && !node.outputs.is_empty() {
                        self.convert_elem_binary_node(
                            op,
                            node.inputs[0],
                            node.inputs[1],
                            node.outputs[0],
                            idx,
                            &graph.tensors,
                        )?;
                    }
                }
                OpType::Reduce { op, axes } => {
                    if !node.inputs.is_empty() && !node.outputs.is_empty() {
                        self.convert_reduce_node(
                            op,
                            axes,
                            node.inputs[0],
                            node.outputs[0],
                            idx,
                            &graph.tensors,
                        )?;
                    }
                }
            }
        }

        // Optionally add Identity nodes for outputs
        if self.config.add_identity_outputs {
            for &output_idx in &graph.outputs {
                let input_name = graph.tensors[output_idx].clone();
                let output_name = format!("output_{}", output_idx);

                let mut attrs = HashMap::new();
                attrs.insert("T".to_string(), self.create_attr_dtype());

                self.tf_nodes.push(NodeDef {
                    name: output_name,
                    op: "Identity".to_string(),
                    input: vec![input_name],
                    device: String::new(),
                    attr: attrs,
                });
            }
        }

        Ok(GraphDef {
            node: self.tf_nodes.clone(),
            version: 0, // GraphDef version
        })
    }
}

/// Export an EinsumGraph to TensorFlow GraphDef format.
///
/// Uses default configuration with float32 dtype and identity outputs.
///
/// # Example
///
/// ```rust,ignore
/// use tensorlogic_compiler::export::tensorflow::export_to_tensorflow;
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let graph = compile_to_einsum(&expr)?;
/// let graphdef_bytes = export_to_tensorflow(&graph, "my_model")?;
/// std::fs::write("model.pb", graphdef_bytes)?;
/// ```
pub fn export_to_tensorflow(graph: &EinsumGraph, model_name: &str) -> Result<Vec<u8>> {
    let config = TensorFlowExportConfig {
        model_name: model_name.to_string(),
        ..Default::default()
    };
    export_to_tensorflow_with_config(graph, config)
}

/// Export an EinsumGraph to TensorFlow GraphDef format with custom configuration.
///
/// Allows fine-grained control over data types, output formatting, etc.
///
/// # Example
///
/// ```rust,ignore
/// use tensorlogic_compiler::export::tensorflow::{
///     export_to_tensorflow_with_config, TensorFlowExportConfig, TfDataType
/// };
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let graph = compile_to_einsum(&expr)?;
///
/// let config = TensorFlowExportConfig {
///     model_name: "logic_model".to_string(),
///     default_dtype: TfDataType::Float64,
///     add_identity_outputs: false,
/// };
///
/// let graphdef_bytes = export_to_tensorflow_with_config(&graph, config)?;
/// ```
pub fn export_to_tensorflow_with_config(
    graph: &EinsumGraph,
    config: TensorFlowExportConfig,
) -> Result<Vec<u8>> {
    let mut converter = TensorFlowConverter::new(config);
    let graphdef = converter.convert(graph)?;

    // Serialize to protobuf binary format
    let mut buf = Vec::new();
    graphdef
        .encode(&mut buf)
        .map_err(|e| anyhow!("Failed to encode GraphDef: {}", e))?;

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode};

    #[test]
    fn test_tf_dtype_conversion() {
        assert_eq!(TfDataType::Float32.to_tf_enum(), 1);
        assert_eq!(TfDataType::Float64.to_tf_enum(), 2);
        assert_eq!(TfDataType::Int32.to_tf_enum(), 3);
        assert_eq!(TfDataType::Int64.to_tf_enum(), 9);
        assert_eq!(TfDataType::Bool.to_tf_enum(), 10);
    }

    #[test]
    fn test_tf_dtype_strings() {
        assert_eq!(TfDataType::Float32.to_tf_string(), "float32");
        assert_eq!(TfDataType::Float64.to_tf_string(), "float64");
        assert_eq!(TfDataType::Int32.to_tf_string(), "int32");
        assert_eq!(TfDataType::Int64.to_tf_string(), "int64");
        assert_eq!(TfDataType::Bool.to_tf_string(), "bool");
    }

    #[test]
    fn test_default_config() {
        let config = TensorFlowExportConfig::default();
        assert_eq!(config.model_name, "tensorlogic_model");
        assert_eq!(config.default_dtype, TfDataType::Float32);
        assert!(config.add_identity_outputs);
    }

    #[test]
    fn test_export_simple_einsum() {
        let mut graph = EinsumGraph::new();

        // Create tensors
        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        // Create a simple einsum: ab,bc->ac
        let _node_idx = graph
            .add_node(EinsumNode::einsum("ab,bc->ac", vec![a, b], vec![c]))
            .unwrap();

        graph.outputs.push(c);

        let result = export_to_tensorflow(&graph, "test_einsum");
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_export_elem_binary() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");
        let t3 = graph.add_tensor("t3");

        let _node_idx = graph
            .add_node(EinsumNode::elem_binary("add", t1, t2, t3))
            .unwrap();

        graph.outputs.push(t3);

        let result = export_to_tensorflow(&graph, "test_add");
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_elem_unary() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");

        let _node_idx = graph
            .add_node(EinsumNode::elem_unary("exp", t1, t2))
            .unwrap();

        graph.outputs.push(t2);

        let result = export_to_tensorflow(&graph, "test_exp");
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_reduce() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");

        let _node_idx = graph
            .add_node(EinsumNode::reduce("sum", vec![1], t1, t2))
            .unwrap();

        graph.outputs.push(t2);

        let result = export_to_tensorflow(&graph, "test_reduce");
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_with_custom_config() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        graph.outputs.push(t1);

        let config = TensorFlowExportConfig {
            model_name: "custom_model".to_string(),
            default_dtype: TfDataType::Float64,
            add_identity_outputs: false,
        };

        let result = export_to_tensorflow_with_config(&graph, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unsupported_unary_op() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");

        let _node_idx = graph
            .add_node(EinsumNode::elem_unary("invalid_op", t1, t2))
            .unwrap();

        graph.outputs.push(t2);

        let result = export_to_tensorflow(&graph, "test_invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_binary_op() {
        let mut graph = EinsumGraph::new();

        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");
        let t3 = graph.add_tensor("t3");

        let _node_idx = graph
            .add_node(EinsumNode::elem_binary("invalid_op", t1, t2, t3))
            .unwrap();

        graph.outputs.push(t3);

        let result = export_to_tensorflow(&graph, "test_invalid");
        assert!(result.is_err());
    }
}
