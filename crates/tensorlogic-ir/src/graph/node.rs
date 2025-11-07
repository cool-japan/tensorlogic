//! Computation nodes in the tensor graph.

use serde::{Deserialize, Serialize};

use crate::error::IrError;
use crate::metadata::Metadata;

use super::{EinsumSpec, OpType};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EinsumNode {
    pub op: OpType,
    pub inputs: Vec<usize>,
    /// Tensor indices that this node produces/writes to.
    /// Most operations produce a single tensor, but some may produce multiple.
    pub outputs: Vec<usize>,
    /// Optional metadata for debugging and provenance tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl EinsumNode {
    pub fn new(spec: impl Into<String>, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        EinsumNode {
            op: OpType::Einsum { spec: spec.into() },
            inputs,
            outputs,
            metadata: None,
        }
    }

    pub fn einsum(spec: impl Into<String>, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        Self::new(spec, inputs, outputs)
    }

    pub fn elem_unary(op: impl Into<String>, input: usize, output: usize) -> Self {
        EinsumNode {
            op: OpType::ElemUnary { op: op.into() },
            inputs: vec![input],
            outputs: vec![output],
            metadata: None,
        }
    }

    pub fn elem_binary(op: impl Into<String>, left: usize, right: usize, output: usize) -> Self {
        EinsumNode {
            op: OpType::ElemBinary { op: op.into() },
            inputs: vec![left, right],
            outputs: vec![output],
            metadata: None,
        }
    }

    pub fn reduce(op: impl Into<String>, axes: Vec<usize>, input: usize, output: usize) -> Self {
        EinsumNode {
            op: OpType::Reduce {
                op: op.into(),
                axes,
            },
            inputs: vec![input],
            outputs: vec![output],
            metadata: None,
        }
    }

    /// Creates a node with automatic output tracking.
    /// The output tensor index should be provided by the caller after calling add_tensor().
    /// This is a convenience method for the common case of single-output operations.
    pub fn with_single_output(
        spec: impl Into<String>,
        inputs: Vec<usize>,
        output_idx: usize,
    ) -> Self {
        Self::new(spec, inputs, vec![output_idx])
    }

    pub fn validate(&self, num_tensors: usize) -> Result<(), IrError> {
        if let OpType::Einsum { spec } = &self.op {
            if spec.is_empty() {
                return Err(IrError::EmptyEinsumSpec);
            }
        }

        for &idx in &self.inputs {
            if idx >= num_tensors {
                return Err(IrError::TensorIndexOutOfBounds {
                    index: idx,
                    max: num_tensors - 1,
                });
            }
        }

        for &idx in &self.outputs {
            if idx >= num_tensors {
                return Err(IrError::TensorIndexOutOfBounds {
                    index: idx,
                    max: num_tensors - 1,
                });
            }
        }

        Ok(())
    }

    /// Get the primary output tensor index (first output).
    /// Most operations produce a single tensor.
    pub fn primary_output(&self) -> Option<usize> {
        self.outputs.first().copied()
    }

    /// Check if this node produces a specific tensor.
    pub fn produces(&self, tensor_idx: usize) -> bool {
        self.outputs.contains(&tensor_idx)
    }

    /// Parse and validate the einsum spec if this is an Einsum operation.
    pub fn parse_einsum_spec(&self) -> Result<Option<EinsumSpec>, IrError> {
        match &self.op {
            OpType::Einsum { spec } => {
                let parsed = EinsumSpec::parse(spec)?;
                parsed.validate_input_count(self.inputs.len())?;
                Ok(Some(parsed))
            }
            _ => Ok(None),
        }
    }

    /// Get a human-readable description of this node's operation.
    pub fn operation_description(&self) -> String {
        match &self.op {
            OpType::Einsum { spec } => format!("Einsum({})", spec),
            OpType::ElemUnary { op } => format!("ElemUnary({})", op),
            OpType::ElemBinary { op } => format!("ElemBinary({})", op),
            OpType::Reduce { op, axes } => format!("Reduce({}, axes={:?})", op, axes),
        }
    }

    /// Attach metadata to this node.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get the metadata if present.
    pub fn get_metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Set the metadata for this node.
    pub fn set_metadata(&mut self, metadata: Metadata) {
        self.metadata = Some(metadata);
    }
}
