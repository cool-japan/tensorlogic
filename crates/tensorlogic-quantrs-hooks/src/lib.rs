//! TL <-> QuantrS2 hooks (PGM/message passing as reductions).
//!
//! This crate provides integration between TensorLogic and probabilistic graphical models (PGMs).
//! It maps belief propagation and other message passing algorithms onto einsum reduction patterns.
//!
//! # Core Concepts
//!
//! - **Factor Graphs**: Convert TLExpr predicates into factors
//! - **Message Passing**: Sum-product and max-product algorithms as tensor operations
//! - **Inference**: Marginalization and conditional queries via reductions
//!
//! # Architecture
//!
//! ```text
//! TLExpr → FactorGraph → MessagePassing → Marginals
//!    ↓         ↓              ↓              ↓
//! Predicates Factors    Einsum Ops    Probabilities
//! ```

mod error;
mod expectation_propagation;
mod factor;
mod graph;
mod inference;
mod junction_tree;
mod linear_chain_crf;
mod message_passing;
mod models;
pub mod parameter_learning;
pub mod quantrs_hooks;
mod sampling;
mod variable_elimination;
mod variational;

pub use error::{PgmError, Result};
pub use expectation_propagation::{ExpectationPropagation, GaussianEP, GaussianSite, Site};
pub use factor::{Factor, FactorOp};
pub use graph::{FactorGraph, FactorNode, VariableNode};
pub use inference::{ConditionalQuery, InferenceEngine, MarginalizationQuery};
pub use junction_tree::{Clique, JunctionTree, JunctionTreeEdge, Separator};
pub use linear_chain_crf::{
    EmissionFeature, FeatureFunction, IdentityFeature, LinearChainCRF, TransitionFeature,
};
pub use message_passing::{
    ConvergenceStats, MaxProductAlgorithm, MessagePassingAlgorithm, SumProductAlgorithm,
};
pub use models::{BayesianNetwork, ConditionalRandomField, HiddenMarkovModel, MarkovRandomField};
pub use parameter_learning::{
    BaumWelchLearner, BayesianEstimator, MaximumLikelihoodEstimator, SimpleHMM,
};
pub use quantrs_hooks::{
    DistributionExport, DistributionMetadata, ModelExport, ModelStatistics, QuantRSAssignment,
    QuantRSDistribution, QuantRSInferenceQuery, QuantRSModelExport, QuantRSParameterLearning,
    QuantRSSamplingHook,
};
pub use sampling::{Assignment, GibbsSampler};
pub use variable_elimination::VariableElimination;
pub use variational::{BetheApproximation, MeanFieldInference, TreeReweightedBP};

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use tensorlogic_ir::TLExpr;

/// Convert a TensorLogic expression to a factor graph.
///
/// This function analyzes the logical structure and creates a factor graph
/// where predicates become factors and quantified variables become nodes.
pub fn expr_to_factor_graph(expr: &TLExpr) -> Result<FactorGraph> {
    let mut graph = FactorGraph::new();

    // Recursively extract factors from expression
    extract_factors(expr, &mut graph)?;

    Ok(graph)
}

/// Extract factors from a TLExpr and add them to the factor graph.
fn extract_factors(expr: &TLExpr, graph: &mut FactorGraph) -> Result<()> {
    match expr {
        TLExpr::Pred { name, args } => {
            // Create a factor from predicate
            let var_names: Vec<String> = args
                .iter()
                .filter_map(|term| match term {
                    tensorlogic_ir::Term::Var(v) => Some(v.clone()),
                    _ => None,
                })
                .collect();

            // Add variables if they don't exist
            for var_name in &var_names {
                if graph.get_variable(var_name).is_none() {
                    graph.add_variable(var_name.clone(), "default".to_string());
                }
            }

            if !var_names.is_empty() {
                graph.add_factor_from_predicate(name, &var_names)?;
            }
        }
        TLExpr::And(left, right) => {
            // Conjunction creates multiple factors
            extract_factors(left, graph)?;
            extract_factors(right, graph)?;
        }
        TLExpr::Exists { var, domain, body } | TLExpr::ForAll { var, domain, body } => {
            // Quantified variables become nodes in the factor graph
            graph.add_variable(var.clone(), domain.clone());
            extract_factors(body, graph)?;
        }
        TLExpr::Imply(premise, conclusion) => {
            // Implication can be represented as factors
            extract_factors(premise, graph)?;
            extract_factors(conclusion, graph)?;
        }
        TLExpr::Not(inner) => {
            // Negation affects factor values
            extract_factors(inner, graph)?;
        }
        _ => {
            // Other expressions may not directly map to factors
        }
    }

    Ok(())
}

/// Perform message passing inference on a factor graph.
///
/// This function runs belief propagation to compute marginal probabilities.
pub fn message_passing_reduce(
    graph: &FactorGraph,
    algorithm: &dyn MessagePassingAlgorithm,
) -> Result<HashMap<String, ArrayD<f64>>> {
    algorithm.run(graph)
}

/// Compute marginal probability for a variable.
///
/// This maps to a reduction operation over all other variables.
pub fn marginalize(
    joint_distribution: &ArrayD<f64>,
    variable_idx: usize,
    axes_to_sum: &[usize],
) -> Result<ArrayD<f64>> {
    use scirs2_core::ndarray::Axis;

    let mut result = joint_distribution.clone();

    // Sum over all axes except the target variable
    for &axis in axes_to_sum.iter().rev() {
        if axis != variable_idx {
            result = result.sum_axis(Axis(axis));
        }
    }

    Ok(result)
}

/// Compute conditional probability P(X | Y = y).
///
/// This slices the joint distribution at the evidence values.
pub fn condition(
    joint_distribution: &ArrayD<f64>,
    evidence: &HashMap<usize, usize>,
) -> Result<ArrayD<f64>> {
    let mut result = joint_distribution.clone();

    // Slice at evidence values
    for (&var_idx, &value) in evidence {
        result = result.index_axis_move(scirs2_core::ndarray::Axis(var_idx), value);
    }

    // Normalize
    let sum: f64 = result.iter().sum();
    if sum > 0.0 {
        result /= sum;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;
    use tensorlogic_ir::Term;

    #[test]
    fn test_expr_to_factor_graph() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let graph = expr_to_factor_graph(&expr).unwrap();
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_marginalize_simple() {
        // 2x2 joint distribution: P(X, Y)
        let joint = Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])
            .unwrap()
            .into_dyn();

        // Marginalize over Y (axis 1) to get P(X)
        let marginal = marginalize(&joint, 0, &[0, 1]).unwrap();

        assert_eq!(marginal.ndim(), 1);
        assert_abs_diff_eq!(marginal.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_condition_simple() {
        // 2x2 joint distribution
        let joint = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();

        // Condition on Y=1: P(X | Y=1)
        let mut evidence = HashMap::new();
        evidence.insert(1, 1);

        let conditional = condition(&joint, &evidence).unwrap();

        // Should have one dimension less
        assert_eq!(conditional.ndim(), 1);
        // Should be normalized
        assert_abs_diff_eq!(conditional.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_factor_graph_construction() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "Domain1".to_string());
        graph.add_variable("y".to_string(), "Domain2".to_string());

        assert_eq!(graph.num_variables(), 2);
    }
}
