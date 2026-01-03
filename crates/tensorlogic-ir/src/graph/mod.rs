//! Tensor computation graphs (EinsumGraph).

pub mod advanced_algorithms;
pub mod advanced_analysis;
pub mod canonicalization;
pub mod constant_folding;
pub mod cost_model;
pub mod dot_export;
mod einsum_spec;
mod einsum_spec_display;
pub mod export;
pub mod fusion;
pub mod layout;
pub mod memory;
mod node;
pub mod optimization;
mod optype;
pub mod parallel;
pub mod pattern;
pub mod pgo;
pub mod schedule;
pub mod tiling;
pub mod transform;
pub mod validation;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use canonicalization::{are_graphs_equivalent, canonical_hash, canonicalize_graph};
pub use dot_export::{export_to_dot, export_to_dot_with_options, DotExportOptions};
pub use einsum_spec::EinsumSpec;
pub use node::EinsumNode;
pub use optimization::{
    eliminate_common_subexpressions, eliminate_dead_code, optimize_graph,
    simplify_identity_operations, OptimizationStats,
};
pub use optype::OpType;
// Public API traits for graph transformation - meant for external use
#[allow(unused_imports)]
pub use transform::{GraphMutVisitor, GraphVisitor};
pub use validation::{
    validate_graph, GraphValidationStats, ValidationError, ValidationErrorKind, ValidationReport,
    ValidationWarning, ValidationWarningKind,
};

use crate::error::IrError;
use crate::metadata::Metadata;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EinsumGraph {
    pub tensors: Vec<String>,
    pub nodes: Vec<EinsumNode>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Metadata for tensors (indexed by tensor index)
    #[serde(default)]
    pub tensor_metadata: HashMap<usize, Metadata>,
}

impl EinsumGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(tensor_cap: usize, node_cap: usize) -> Self {
        EinsumGraph {
            tensors: Vec::with_capacity(tensor_cap),
            nodes: Vec::with_capacity(node_cap),
            inputs: Vec::new(),
            outputs: Vec::new(),
            tensor_metadata: HashMap::new(),
        }
    }

    pub fn add_tensor(&mut self, name: impl Into<String>) -> usize {
        let idx = self.tensors.len();
        self.tensors.push(name.into());
        idx
    }

    pub fn add_node(&mut self, node: EinsumNode) -> Result<usize, IrError> {
        node.validate(self.tensors.len())?;
        let idx = self.nodes.len();
        self.nodes.push(node);
        Ok(idx)
    }

    pub fn add_input(&mut self, tensor_idx: usize) -> Result<(), IrError> {
        if tensor_idx >= self.tensors.len() {
            return Err(IrError::TensorIndexOutOfBounds {
                index: tensor_idx,
                max: self.tensors.len() - 1,
            });
        }
        self.inputs.push(tensor_idx);
        Ok(())
    }

    pub fn add_output(&mut self, tensor_idx: usize) -> Result<(), IrError> {
        if tensor_idx >= self.tensors.len() {
            return Err(IrError::OutputIndexOutOfBounds {
                index: tensor_idx,
                max: self.tensors.len() - 1,
            });
        }
        self.outputs.push(tensor_idx);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), IrError> {
        for (idx, node) in self.nodes.iter().enumerate() {
            node.validate(self.tensors.len())
                .map_err(|e| IrError::NodeValidation {
                    node: idx,
                    message: e.to_string(),
                })?;
        }

        for &out_idx in &self.outputs {
            if out_idx >= self.tensors.len() {
                return Err(IrError::OutputIndexOutOfBounds {
                    index: out_idx,
                    max: self.tensors.len() - 1,
                });
            }
        }

        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty() && self.nodes.is_empty()
    }

    /// Add metadata for a tensor.
    pub fn add_tensor_metadata(&mut self, tensor_idx: usize, metadata: Metadata) {
        self.tensor_metadata.insert(tensor_idx, metadata);
    }

    /// Get metadata for a tensor if it exists.
    pub fn get_tensor_metadata(&self, tensor_idx: usize) -> Option<&Metadata> {
        self.tensor_metadata.get(&tensor_idx)
    }

    /// Add a tensor with metadata.
    pub fn add_tensor_with_metadata(
        &mut self,
        name: impl Into<String>,
        metadata: Metadata,
    ) -> usize {
        let idx = self.add_tensor(name);
        self.add_tensor_metadata(idx, metadata);
        idx
    }
}
