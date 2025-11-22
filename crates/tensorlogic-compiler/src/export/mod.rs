//! Export compiled TensorLogic graphs to various formats.
//!
//! This module provides functionality to export `EinsumGraph` instances to different
//! interchange formats for execution on various backends.
//!
//! # Supported Formats
//!
//! - **ONNX** (`onnx` feature): Export to ONNX format for use with ONNX Runtime,
//!   PyTorch, TensorFlow, and other ONNX-compatible frameworks.
//! - **TensorFlow GraphDef** (`tensorflow` feature): Export to TensorFlow GraphDef format
//!   for execution within TensorFlow runtime and SavedModel workflows.
//! - **PyTorch Code** (`pytorch` feature): Generate PyTorch nn.Module Python code
//!   for integration with PyTorch workflows and TorchScript compilation.
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_compiler::export::onnx::export_to_onnx;
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
//! let graph = compile_to_einsum(&expr)?;
//!
//! // Export to ONNX
//! let onnx_bytes = export_to_onnx(&graph, "logic_model")?;
//! std::fs::write("model.onnx", onnx_bytes)?;
//!
//! // Export to TensorFlow
//! #[cfg(feature = "tensorflow")]
//! {
//!     use tensorlogic_compiler::export::tensorflow::export_to_tensorflow;
//!     let tf_bytes = export_to_tensorflow(&graph, "logic_model")?;
//!     std::fs::write("model.pb", tf_bytes)?;
//! }
//!
//! // Export to PyTorch
//! #[cfg(feature = "pytorch")]
//! {
//!     use tensorlogic_compiler::export::pytorch::export_to_pytorch;
//!     let pytorch_code = export_to_pytorch(&graph, "LogicModel")?;
//!     std::fs::write("model.py", pytorch_code)?;
//! }
//! ```

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "pytorch")]
pub mod pytorch;

#[cfg(feature = "tensorflow")]
pub mod tensorflow;
