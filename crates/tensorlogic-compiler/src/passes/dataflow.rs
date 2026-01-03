//! Dataflow analysis for logical expressions and einsum graphs.
//!
//! This module provides dataflow analysis passes that track how data flows
//! through expressions and computation graphs. These analyses enable powerful
//! optimizations and help identify opportunities for parallelization.
//!
//! # Overview
//!
//! Dataflow analysis is a fundamental compiler technique that tracks:
//! - **Reaching definitions**: Which variable assignments reach each point
//! - **Live variables**: Which variables are used after each point
//! - **Available expressions**: Which expressions have been computed
//! - **Use-def chains**: Relationships between variable uses and definitions
//!
//! # Applications
//!
//! - Dead code elimination
//! - Common subexpression elimination
//! - Register allocation
//! - Constant propagation
//! - Loop optimization
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::passes::analyze_dataflow;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::and(
//!     TLExpr::pred("P", vec![Term::var("x")]),
//!     TLExpr::pred("Q", vec![Term::var("x")]),
//! );
//!
//! let analysis = analyze_dataflow(&expr);
//! println!("Live variables: {:?}", analysis.live_variables);
//! ```

use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{EinsumGraph, TLExpr, Term};

/// Result of dataflow analysis on an expression.
#[derive(Debug, Clone)]
pub struct DataflowAnalysis {
    /// Variables that are live (may be used later) at each expression
    pub live_variables: HashMap<String, HashSet<String>>,
    /// Reaching definitions for each variable
    pub reaching_defs: HashMap<String, HashSet<String>>,
    /// Available expressions at each program point
    pub available_exprs: HashSet<String>,
    /// Use-def chains mapping uses to their definitions
    pub use_def_chains: HashMap<String, Vec<String>>,
    /// Def-use chains mapping definitions to their uses
    pub def_use_chains: HashMap<String, Vec<String>>,
}

impl DataflowAnalysis {
    /// Create a new empty dataflow analysis.
    pub fn new() -> Self {
        Self {
            live_variables: HashMap::new(),
            reaching_defs: HashMap::new(),
            available_exprs: HashSet::new(),
            use_def_chains: HashMap::new(),
            def_use_chains: HashMap::new(),
        }
    }

    /// Check if a variable is live at a given point.
    pub fn is_live(&self, expr_id: &str, var: &str) -> bool {
        self.live_variables
            .get(expr_id)
            .map(|vars| vars.contains(var))
            .unwrap_or(false)
    }

    /// Get all live variables at a given point.
    pub fn get_live_vars(&self, expr_id: &str) -> HashSet<String> {
        self.live_variables
            .get(expr_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get reaching definitions for a variable.
    pub fn get_reaching_defs(&self, var: &str) -> HashSet<String> {
        self.reaching_defs.get(var).cloned().unwrap_or_default()
    }

    /// Check if an expression is available at a point.
    pub fn is_available(&self, expr: &str) -> bool {
        self.available_exprs.contains(expr)
    }

    /// Get use-def chain for a variable use.
    pub fn get_use_def_chain(&self, var: &str) -> Vec<String> {
        self.use_def_chains.get(var).cloned().unwrap_or_default()
    }

    /// Get def-use chain for a variable definition.
    pub fn get_def_use_chain(&self, var: &str) -> Vec<String> {
        self.def_use_chains.get(var).cloned().unwrap_or_default()
    }
}

impl Default for DataflowAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for dataflow analysis.
#[derive(Debug, Clone)]
pub struct DataflowConfig {
    /// Compute live variable analysis
    pub compute_live_vars: bool,
    /// Compute reaching definitions
    pub compute_reaching_defs: bool,
    /// Compute available expressions
    pub compute_available_exprs: bool,
    /// Compute use-def chains
    pub compute_use_def_chains: bool,
}

impl Default for DataflowConfig {
    fn default() -> Self {
        Self {
            compute_live_vars: true,
            compute_reaching_defs: true,
            compute_available_exprs: true,
            compute_use_def_chains: true,
        }
    }
}

/// Perform dataflow analysis on a logical expression.
pub fn analyze_dataflow(expr: &TLExpr) -> DataflowAnalysis {
    analyze_dataflow_with_config(expr, &DataflowConfig::default())
}

/// Perform dataflow analysis with custom configuration.
pub fn analyze_dataflow_with_config(expr: &TLExpr, config: &DataflowConfig) -> DataflowAnalysis {
    let mut analysis = DataflowAnalysis::new();

    if config.compute_live_vars {
        compute_live_variables(expr, &mut analysis);
    }

    if config.compute_reaching_defs {
        compute_reaching_definitions(expr, &mut analysis);
    }

    if config.compute_available_exprs {
        compute_available_expressions(expr, &mut analysis);
    }

    if config.compute_use_def_chains {
        compute_use_def_chains(expr, &mut analysis);
    }

    analysis
}

/// Compute live variable analysis.
///
/// A variable is live at a point if it may be used after that point.
fn compute_live_variables(expr: &TLExpr, analysis: &mut DataflowAnalysis) {
    let expr_id = format!("{:?}", expr as *const _);
    let mut live = HashSet::new();

    // Collect variables used in this expression
    match expr {
        TLExpr::Pred { args, .. } => {
            for arg in args {
                if let Term::Var(v) = arg {
                    live.insert(v.clone());
                }
            }
        }
        TLExpr::And(lhs, rhs) | TLExpr::Or(lhs, rhs) | TLExpr::Imply(lhs, rhs) => {
            // Union of live variables from both branches
            compute_live_variables(lhs, analysis);
            compute_live_variables(rhs, analysis);

            let lhs_live = analysis.get_live_vars(&format!("{:?}", lhs.as_ref() as *const _));
            let rhs_live = analysis.get_live_vars(&format!("{:?}", rhs.as_ref() as *const _));
            live.extend(lhs_live);
            live.extend(rhs_live);
        }
        TLExpr::Not(inner) => {
            compute_live_variables(inner, analysis);
            let inner_live = analysis.get_live_vars(&format!("{:?}", inner.as_ref() as *const _));
            live.extend(inner_live);
        }
        TLExpr::Exists { var, body, .. } | TLExpr::ForAll { var, body, .. } => {
            compute_live_variables(body, analysis);
            let mut body_live = analysis.get_live_vars(&format!("{:?}", body.as_ref() as *const _));

            // Remove the bound variable
            body_live.remove(var);
            live.extend(body_live);
        }
        TLExpr::Let { var, value, body } => {
            compute_live_variables(value, analysis);
            compute_live_variables(body, analysis);

            let mut body_live = analysis.get_live_vars(&format!("{:?}", body.as_ref() as *const _));
            let value_live = analysis.get_live_vars(&format!("{:?}", value.as_ref() as *const _));

            // Variable is defined here, remove from live set
            body_live.remove(var);
            live.extend(body_live);
            live.extend(value_live);
        }
        _ => {
            // For other expressions, just collect free variables
            live.extend(expr.free_vars());
        }
    }

    analysis.live_variables.insert(expr_id, live);
}

/// Compute reaching definitions analysis.
///
/// A definition reaches a point if it may be the most recent assignment
/// to a variable at that point.
fn compute_reaching_definitions(expr: &TLExpr, analysis: &mut DataflowAnalysis) {
    match expr {
        TLExpr::Let { var, value, body } => {
            // This is a definition of 'var'
            let def_id = format!("let_{}", var);
            analysis
                .reaching_defs
                .entry(var.clone())
                .or_default()
                .insert(def_id);

            compute_reaching_definitions(value, analysis);
            compute_reaching_definitions(body, analysis);
        }
        TLExpr::Exists { var, body, .. } | TLExpr::ForAll { var, body, .. } => {
            // Quantifier introduces a new scope for var
            let def_id = format!("quant_{}", var);
            analysis
                .reaching_defs
                .entry(var.clone())
                .or_default()
                .insert(def_id);

            compute_reaching_definitions(body, analysis);
        }
        TLExpr::And(lhs, rhs) | TLExpr::Or(lhs, rhs) | TLExpr::Imply(lhs, rhs) => {
            compute_reaching_definitions(lhs, analysis);
            compute_reaching_definitions(rhs, analysis);
        }
        TLExpr::Not(inner) => {
            compute_reaching_definitions(inner, analysis);
        }
        _ => {
            // Leaf expressions don't introduce definitions
        }
    }
}

/// Compute available expressions analysis.
///
/// An expression is available at a point if it has been computed and
/// not invalidated since.
fn compute_available_expressions(expr: &TLExpr, analysis: &mut DataflowAnalysis) {
    let expr_str = format!("{:?}", expr);

    match expr {
        TLExpr::Pred { .. } | TLExpr::Constant(_) => {
            // Atomic expressions are always available
            analysis.available_exprs.insert(expr_str);
        }
        TLExpr::And(lhs, rhs) | TLExpr::Or(lhs, rhs) | TLExpr::Imply(lhs, rhs) => {
            compute_available_expressions(lhs, analysis);
            compute_available_expressions(rhs, analysis);

            // This expression is available if both operands are
            analysis.available_exprs.insert(expr_str);
        }
        TLExpr::Not(inner) => {
            compute_available_expressions(inner, analysis);
            analysis.available_exprs.insert(expr_str);
        }
        TLExpr::Let { value, body, .. } => {
            compute_available_expressions(value, analysis);
            compute_available_expressions(body, analysis);
        }
        _ => {
            // Other expressions may be available
            analysis.available_exprs.insert(expr_str);
        }
    }
}

/// Compute use-def chains.
///
/// Maps each variable use to the definitions that may reach it.
fn compute_use_def_chains(expr: &TLExpr, analysis: &mut DataflowAnalysis) {
    // First compute reaching definitions
    compute_reaching_definitions(expr, analysis);

    // Then build use-def chains by linking uses to their reaching defs
    collect_uses(expr, analysis);
}

/// Collect variable uses and link them to definitions.
fn collect_uses(expr: &TLExpr, analysis: &mut DataflowAnalysis) {
    match expr {
        TLExpr::Pred { args, .. } => {
            for arg in args {
                if let Term::Var(v) = arg {
                    // Link this use to its reaching definitions
                    let defs = analysis.get_reaching_defs(v);
                    analysis
                        .use_def_chains
                        .entry(v.clone())
                        .or_default()
                        .extend(defs.iter().cloned());

                    // Also update def-use chains
                    for def in defs {
                        analysis
                            .def_use_chains
                            .entry(def)
                            .or_default()
                            .push(v.clone());
                    }
                }
            }
        }
        TLExpr::And(lhs, rhs) | TLExpr::Or(lhs, rhs) | TLExpr::Imply(lhs, rhs) => {
            collect_uses(lhs, analysis);
            collect_uses(rhs, analysis);
        }
        TLExpr::Not(inner) => {
            collect_uses(inner, analysis);
        }
        TLExpr::Let { value, body, .. } => {
            collect_uses(value, analysis);
            collect_uses(body, analysis);
        }
        TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
            collect_uses(body, analysis);
        }
        _ => {}
    }
}

/// Dataflow analysis for einsum graphs.
#[derive(Debug, Clone)]
pub struct GraphDataflow {
    /// Live tensors at each node
    pub live_tensors: HashMap<usize, HashSet<usize>>,
    /// Tensor dependencies
    pub dependencies: HashMap<usize, HashSet<usize>>,
    /// Reverse dependencies (uses)
    pub uses: HashMap<usize, HashSet<usize>>,
}

impl GraphDataflow {
    /// Create a new graph dataflow analysis.
    pub fn new() -> Self {
        Self {
            live_tensors: HashMap::new(),
            dependencies: HashMap::new(),
            uses: HashMap::new(),
        }
    }

    /// Check if a tensor is live at a node.
    pub fn is_tensor_live(&self, node: usize, tensor: usize) -> bool {
        self.live_tensors
            .get(&node)
            .map(|tensors| tensors.contains(&tensor))
            .unwrap_or(false)
    }

    /// Get dependencies of a tensor.
    pub fn get_dependencies(&self, tensor: usize) -> HashSet<usize> {
        self.dependencies.get(&tensor).cloned().unwrap_or_default()
    }

    /// Get uses of a tensor.
    pub fn get_uses(&self, tensor: usize) -> HashSet<usize> {
        self.uses.get(&tensor).cloned().unwrap_or_default()
    }
}

impl Default for GraphDataflow {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze dataflow in an einsum graph.
pub fn analyze_graph_dataflow(graph: &EinsumGraph) -> GraphDataflow {
    let mut analysis = GraphDataflow::new();

    // Compute dependencies
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output in &node.outputs {
            let mut deps = HashSet::new();
            deps.extend(&node.inputs);

            analysis.dependencies.insert(output, deps);

            // Update reverse dependencies (uses)
            for &input in &node.inputs {
                analysis.uses.entry(input).or_default().insert(node_idx);
            }
        }
    }

    // Compute live tensors (backward analysis)
    let mut live: HashSet<usize> = HashSet::new();
    live.extend(&graph.outputs);

    for (node_idx, node) in graph.nodes.iter().enumerate().rev() {
        // Tensors are live if they're used by later nodes or are outputs
        let node_live: HashSet<usize> = node
            .outputs
            .iter()
            .filter(|&&t| live.contains(&t) || graph.outputs.contains(&t))
            .copied()
            .collect();

        if !node_live.is_empty() {
            // Add inputs to live set
            live.extend(&node.inputs);
        }

        analysis.live_tensors.insert(node_idx, node_live);
    }

    analysis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_variables_simple() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let analysis = analyze_dataflow(&expr);

        // x should be live in the predicate
        assert!(!analysis.live_variables.is_empty());
    }

    #[test]
    fn test_live_variables_and() {
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y")]),
        );

        let analysis = analyze_dataflow(&expr);

        // Both x and y should be live
        assert!(!analysis.live_variables.is_empty());
    }

    #[test]
    fn test_reaching_definitions_let() {
        let expr = TLExpr::Let {
            var: "x".to_string(),
            value: Box::new(TLExpr::Constant(1.0)),
            body: Box::new(TLExpr::pred("P", vec![Term::var("x")])),
        };

        let analysis = analyze_dataflow(&expr);

        // Should have a reaching definition for x
        assert!(analysis.reaching_defs.contains_key("x"));
    }

    #[test]
    fn test_quantifier_binding() {
        let expr = TLExpr::exists("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

        let analysis = analyze_dataflow(&expr);

        // x is bound by exists, so it shouldn't be in the live set of the outer expression
        let expr_id = format!("{:?}", &expr as *const _);
        let live = analysis.get_live_vars(&expr_id);
        assert!(!live.contains("x"));
    }

    #[test]
    fn test_available_expressions() {
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        );

        let analysis = analyze_dataflow(&expr);

        // Should have available expressions
        assert!(!analysis.available_exprs.is_empty());
    }

    #[test]
    fn test_graph_dataflow() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("t0");
        let t1 = graph.add_tensor("t1");

        let node = graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();
        graph.add_output(t1).unwrap();

        let analysis = analyze_graph_dataflow(&graph);

        // t0 should be a dependency of t1
        let deps = analysis.get_dependencies(t1);
        assert!(deps.contains(&t0));

        // t1 should be live
        assert!(analysis.is_tensor_live(node, t1));
    }

    #[test]
    fn test_dataflow_config() {
        let config = DataflowConfig {
            compute_live_vars: true,
            compute_reaching_defs: false,
            compute_available_exprs: false,
            compute_use_def_chains: false,
        };

        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let analysis = analyze_dataflow_with_config(&expr, &config);

        // Only live variables should be computed
        assert!(!analysis.live_variables.is_empty());
    }

    #[test]
    fn test_use_def_chains() {
        let expr = TLExpr::Let {
            var: "x".to_string(),
            value: Box::new(TLExpr::Constant(1.0)),
            body: Box::new(TLExpr::pred("P", vec![Term::var("x")])),
        };

        let analysis = analyze_dataflow(&expr);

        // Should have use-def chains for x
        assert!(!analysis.use_def_chains.is_empty() || !analysis.def_use_chains.is_empty());
    }

    #[test]
    fn test_graph_dependencies() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("t0");
        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");

        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();
        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("log", t1, t2))
            .unwrap();

        let analysis = analyze_graph_dataflow(&graph);

        // t2 depends on t1, t1 depends on t0
        assert!(analysis.get_dependencies(t1).contains(&t0));
        assert!(analysis.get_dependencies(t2).contains(&t1));
    }

    #[test]
    fn test_dataflow_analysis_default() {
        let analysis = DataflowAnalysis::new();
        assert!(analysis.live_variables.is_empty());
        assert!(analysis.reaching_defs.is_empty());
        assert!(analysis.available_exprs.is_empty());
    }
}
