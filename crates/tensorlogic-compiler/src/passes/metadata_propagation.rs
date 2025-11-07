//! Metadata propagation for provenance tracking and debugging.
//!
//! This module provides utilities for attaching metadata to compiled tensor
//! graphs, enabling better debugging, provenance tracking, and understanding
//! of the compilation process.

use std::collections::HashMap;
use tensorlogic_ir::{EinsumGraph, EinsumNode, Metadata, TLExpr};

use crate::CompilerContext;

/// Metadata builder for tracking compilation provenance
pub struct MetadataBuilder {
    /// Current source file being compiled
    source_file: Option<String>,
    /// Current rule ID being compiled
    rule_id: Option<String>,
    /// Counter for generating unique rule IDs
    rule_counter: usize,
}

impl MetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        Self {
            source_file: None,
            rule_id: None,
            rule_counter: 0,
        }
    }

    /// Set the current source file
    pub fn with_source_file(mut self, file: impl Into<String>) -> Self {
        self.source_file = Some(file.into());
        self
    }

    /// Set the current rule ID
    pub fn with_rule_id(mut self, rule_id: impl Into<String>) -> Self {
        self.rule_id = Some(rule_id.into());
        self
    }

    /// Generate a fresh rule ID
    pub fn fresh_rule_id(&mut self) -> String {
        let id = format!("rule_{}", self.rule_counter);
        self.rule_counter += 1;
        id
    }

    /// Create metadata for a predicate
    pub fn predicate_metadata(&mut self, name: &str, args: &[String]) -> Metadata {
        let mut meta = Metadata::new().with_name(format!("predicate:{}", name));

        if let Some(ref file) = self.source_file {
            meta = meta.with_attribute("source_file", file.clone());
        }

        if let Some(ref rule) = self.rule_id {
            meta = meta.with_attribute("rule_id", rule.clone());
        }

        meta = meta.with_attribute("predicate_name", name.to_string());
        meta = meta.with_attribute("arity", args.len().to_string());

        for (i, arg) in args.iter().enumerate() {
            meta = meta.with_attribute(format!("arg_{}", i), arg.clone());
        }

        meta
    }

    /// Create metadata for a logical operation
    pub fn logic_op_metadata(&mut self, op_type: &str, operand_count: usize) -> Metadata {
        let mut meta = Metadata::new().with_name(format!("logic_op:{}", op_type));

        if let Some(ref file) = self.source_file {
            meta = meta.with_attribute("source_file", file.clone());
        }

        if let Some(ref rule) = self.rule_id {
            meta = meta.with_attribute("rule_id", rule.clone());
        }

        meta = meta.with_attribute("operation", op_type.to_string());
        meta = meta.with_attribute("operand_count", operand_count.to_string());

        meta
    }

    /// Create metadata for a quantifier
    pub fn quantifier_metadata(
        &mut self,
        quantifier_type: &str,
        var: &str,
        domain: &str,
    ) -> Metadata {
        let mut meta = Metadata::new().with_name(format!("quantifier:{}", quantifier_type));

        if let Some(ref file) = self.source_file {
            meta = meta.with_attribute("source_file", file.clone());
        }

        if let Some(ref rule) = self.rule_id {
            meta = meta.with_attribute("rule_id", rule.clone());
        }

        meta = meta.with_attribute("quantifier", quantifier_type.to_string());
        meta = meta.with_attribute("variable", var.to_string());
        meta = meta.with_attribute("domain", domain.to_string());

        meta
    }

    /// Create metadata from TLExpr
    pub fn from_expr(&mut self, expr: &TLExpr) -> Metadata {
        match expr {
            TLExpr::Pred { name, args } => {
                let arg_names: Vec<String> = args.iter().map(|t| format!("{:?}", t)).collect();
                self.predicate_metadata(name, &arg_names)
            }
            TLExpr::And(_, _) => self.logic_op_metadata("AND", 2),
            TLExpr::Or(_, _) => self.logic_op_metadata("OR", 2),
            TLExpr::Not(_) => self.logic_op_metadata("NOT", 1),
            TLExpr::Imply(_, _) => self.logic_op_metadata("IMPLY", 2),
            TLExpr::Exists { var, domain, .. } => self.quantifier_metadata("EXISTS", var, domain),
            TLExpr::ForAll { var, domain, .. } => self.quantifier_metadata("FORALL", var, domain),
            TLExpr::Constant(_) => Metadata::new()
                .with_name("constant")
                .with_attribute("type", "constant"),
            _ => Metadata::new().with_name("expression"),
        }
    }
}

impl Default for MetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagate metadata through a compiled graph
pub fn propagate_metadata(
    graph: &mut EinsumGraph,
    ctx: &CompilerContext,
    _builder: &mut MetadataBuilder,
) {
    // Collect metadata to add (to avoid borrowing issues)
    let mut metadata_to_add: Vec<(usize, Metadata)> = Vec::new();

    // Add domain metadata to input tensors
    for (tensor_idx, tensor_name) in graph.tensors.iter().enumerate() {
        if graph.inputs.contains(&tensor_idx) {
            // Check if this tensor corresponds to a predicate
            if let Some(domain_name) = ctx.var_to_domain.values().find(|d| {
                tensor_name.starts_with(&format!("{}_", d))
                    || tensor_name.contains(&format!("_{}_", d))
            }) {
                let mut meta = Metadata::new()
                    .with_name(format!("input_tensor:{}", tensor_name))
                    .with_attribute("domain", domain_name.clone())
                    .with_attribute("tensor_type", "input");

                if let Some(domain_info) = ctx.domains.get(domain_name) {
                    meta = meta.with_attribute("cardinality", domain_info.cardinality.to_string());
                }

                metadata_to_add.push((tensor_idx, meta));
            }
        }
    }

    // Add domain information as graph-level metadata
    for (domain_name, domain_info) in &ctx.domains {
        // This could be stored as a special metadata attribute on the graph
        // For now, we'll add it as metadata on output tensors if they relate to this domain
        for &output_idx in &graph.outputs {
            if let Some(tensor_name) = graph.tensors.get(output_idx) {
                if tensor_name.contains(domain_name) {
                    let meta = Metadata::new()
                        .with_name(format!("output_tensor:{}", tensor_name))
                        .with_attribute("domain", domain_name.clone())
                        .with_attribute("cardinality", domain_info.cardinality.to_string())
                        .with_attribute("tensor_type", "output");

                    metadata_to_add.push((output_idx, meta));
                }
            }
        }
    }

    // Add all collected metadata
    for (idx, meta) in metadata_to_add {
        graph.add_tensor_metadata(idx, meta);
    }
}

/// Enhanced compilation result with metadata
pub struct MetadataCompilationResult {
    /// The compiled graph
    pub graph: EinsumGraph,
    /// Metadata builder used during compilation
    pub builder: MetadataBuilder,
    /// Mapping from expression to node indices
    pub expr_to_nodes: HashMap<String, Vec<usize>>,
}

impl MetadataCompilationResult {
    /// Create a new result
    pub fn new(graph: EinsumGraph, builder: MetadataBuilder) -> Self {
        Self {
            graph,
            builder,
            expr_to_nodes: HashMap::new(),
        }
    }

    /// Record that an expression was compiled to specific nodes
    pub fn record_expression(&mut self, expr_id: impl Into<String>, node_indices: Vec<usize>) {
        self.expr_to_nodes.insert(expr_id.into(), node_indices);
    }

    /// Get nodes for an expression
    pub fn get_nodes_for_expr(&self, expr_id: &str) -> Option<&[usize]> {
        self.expr_to_nodes.get(expr_id).map(|v| v.as_slice())
    }
}

/// Helper to attach metadata to nodes based on expression type
pub fn attach_expr_metadata(node: &mut EinsumNode, expr: &TLExpr, builder: &mut MetadataBuilder) {
    let metadata = builder.from_expr(expr);
    node.set_metadata(metadata);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_metadata_builder_new() {
        let builder = MetadataBuilder::new();
        assert!(builder.source_file.is_none());
        assert!(builder.rule_id.is_none());
        assert_eq!(builder.rule_counter, 0);
    }

    #[test]
    fn test_metadata_builder_with_source_file() {
        let builder = MetadataBuilder::new().with_source_file("test.tl");
        assert_eq!(builder.source_file, Some("test.tl".to_string()));
    }

    #[test]
    fn test_metadata_builder_fresh_rule_id() {
        let mut builder = MetadataBuilder::new();
        let id1 = builder.fresh_rule_id();
        let id2 = builder.fresh_rule_id();
        assert_eq!(id1, "rule_0");
        assert_eq!(id2, "rule_1");
    }

    #[test]
    fn test_predicate_metadata() {
        let mut builder = MetadataBuilder::new()
            .with_source_file("test.tl")
            .with_rule_id("rule_1");

        let meta = builder.predicate_metadata("knows", &["x".to_string(), "y".to_string()]);

        assert_eq!(meta.name, Some("predicate:knows".to_string()));
        assert_eq!(meta.get_attribute("predicate_name"), Some("knows"));
        assert_eq!(meta.get_attribute("arity"), Some("2"));
        assert_eq!(meta.get_attribute("source_file"), Some("test.tl"));
        assert_eq!(meta.get_attribute("rule_id"), Some("rule_1"));
    }

    #[test]
    fn test_logic_op_metadata() {
        let mut builder = MetadataBuilder::new();
        let meta = builder.logic_op_metadata("AND", 2);

        assert_eq!(meta.name, Some("logic_op:AND".to_string()));
        assert_eq!(meta.get_attribute("operation"), Some("AND"));
        assert_eq!(meta.get_attribute("operand_count"), Some("2"));
    }

    #[test]
    fn test_quantifier_metadata() {
        let mut builder = MetadataBuilder::new();
        let meta = builder.quantifier_metadata("EXISTS", "x", "Person");

        assert_eq!(meta.name, Some("quantifier:EXISTS".to_string()));
        assert_eq!(meta.get_attribute("quantifier"), Some("EXISTS"));
        assert_eq!(meta.get_attribute("variable"), Some("x"));
        assert_eq!(meta.get_attribute("domain"), Some("Person"));
    }

    #[test]
    fn test_from_expr_predicate() {
        let mut builder = MetadataBuilder::new();
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let meta = builder.from_expr(&expr);

        assert_eq!(meta.name, Some("predicate:knows".to_string()));
        assert_eq!(meta.get_attribute("predicate_name"), Some("knows"));
    }

    #[test]
    fn test_from_expr_and() {
        let mut builder = MetadataBuilder::new();
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );
        let meta = builder.from_expr(&expr);

        assert_eq!(meta.name, Some("logic_op:AND".to_string()));
        assert_eq!(meta.get_attribute("operation"), Some("AND"));
    }

    #[test]
    fn test_from_expr_exists() {
        let mut builder = MetadataBuilder::new();
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        );
        let meta = builder.from_expr(&expr);

        assert_eq!(meta.name, Some("quantifier:EXISTS".to_string()));
        assert_eq!(meta.get_attribute("quantifier"), Some("EXISTS"));
        assert_eq!(meta.get_attribute("variable"), Some("x"));
        assert_eq!(meta.get_attribute("domain"), Some("Person"));
    }

    #[test]
    fn test_propagate_metadata_with_domains() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        ctx.bind_var("x", "Person").unwrap();

        let mut graph = EinsumGraph::new();
        let tensor_idx = graph.add_tensor("Person_x");
        graph.inputs.push(tensor_idx);

        let mut builder = MetadataBuilder::new();
        propagate_metadata(&mut graph, &ctx, &mut builder);

        // Check that metadata was added
        let meta = graph.get_tensor_metadata(tensor_idx);
        assert!(meta.is_some());
    }

    #[test]
    fn test_metadata_compilation_result() {
        let graph = EinsumGraph::new();
        let builder = MetadataBuilder::new();
        let mut result = MetadataCompilationResult::new(graph, builder);

        result.record_expression("expr_1", vec![0, 1, 2]);
        assert_eq!(result.get_nodes_for_expr("expr_1"), Some(&[0, 1, 2][..]));
        assert_eq!(result.get_nodes_for_expr("expr_2"), None);
    }

    #[test]
    fn test_attach_expr_metadata() {
        let mut builder = MetadataBuilder::new();
        let mut node = EinsumNode::new("ab->a", vec![0], vec![1]);
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        attach_expr_metadata(&mut node, &expr, &mut builder);

        let meta = node.get_metadata();
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().get_attribute("predicate_name"), Some("knows"));
    }
}
