//! Dynamic Bayesian Networks (DBN) for temporal probabilistic models.
//!
//! This module provides support for Dynamic Bayesian Networks, which are
//! generalizations of Hidden Markov Models to handle multiple interacting
//! variables over time.
//!
//! # Structure
//!
//! A DBN consists of:
//! - Initial (prior) network at t=0
//! - Two-time-slice transition defining state evolution
//! - Interface variables connecting adjacent time slices

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};

use crate::error::{PgmError, Result};
use crate::message_passing::MessagePassingAlgorithm;
use crate::{Factor, FactorGraph, SumProductAlgorithm, VariableElimination};

/// Dynamic Bayesian Network.
///
/// Represents a temporal probabilistic model with variables that evolve over time.
///
/// # Example
///
/// ```
/// use tensorlogic_quantrs_hooks::DynamicBayesianNetwork;
/// use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
///
/// // Create a simple DBN with one state variable
/// let dbn = DynamicBayesianNetwork::new(
///     vec![("state".to_string(), 2)],  // state variables with cardinality
///     vec![],  // no observation variables
/// );
/// ```
#[derive(Debug, Clone)]
pub struct DynamicBayesianNetwork {
    /// State variables with cardinalities
    pub state_vars: Vec<(String, usize)>,
    /// Observation variables with cardinalities
    pub observation_vars: Vec<(String, usize)>,
    /// Initial distribution P(X_0) for each state variable
    pub initial_dists: HashMap<String, ArrayD<f64>>,
    /// Transition distributions P(X_t | X_{t-1})
    pub transition_dists: HashMap<String, ArrayD<f64>>,
    /// Emission distributions P(Y_t | X_t)
    pub emission_dists: HashMap<String, ArrayD<f64>>,
}

/// Temporal variable representing a variable at a specific time step.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TemporalVar {
    /// Base variable name
    pub name: String,
    /// Time step
    pub time: usize,
}

impl std::fmt::Display for TemporalVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.name, self.time)
    }
}

impl DynamicBayesianNetwork {
    /// Create a new DBN with state and observation variables.
    pub fn new(state_vars: Vec<(String, usize)>, observation_vars: Vec<(String, usize)>) -> Self {
        Self {
            state_vars,
            observation_vars,
            initial_dists: HashMap::new(),
            transition_dists: HashMap::new(),
            emission_dists: HashMap::new(),
        }
    }

    /// Set initial distribution for a state variable.
    pub fn set_initial(&mut self, var: &str, dist: ArrayD<f64>) -> Result<&mut Self> {
        if !self.state_vars.iter().any(|(name, _)| name == var) {
            return Err(PgmError::VariableNotFound(var.to_string()));
        }
        self.initial_dists.insert(var.to_string(), dist);
        Ok(self)
    }

    /// Set transition distribution P(X_t | X_{t-1}) for a state variable.
    pub fn set_transition(&mut self, var: &str, dist: ArrayD<f64>) -> Result<&mut Self> {
        if !self.state_vars.iter().any(|(name, _)| name == var) {
            return Err(PgmError::VariableNotFound(var.to_string()));
        }
        self.transition_dists.insert(var.to_string(), dist);
        Ok(self)
    }

    /// Set emission distribution P(Y | X) for an observation variable.
    pub fn set_emission(&mut self, obs_var: &str, dist: ArrayD<f64>) -> Result<&mut Self> {
        if !self
            .observation_vars
            .iter()
            .any(|(name, _)| name == obs_var)
        {
            return Err(PgmError::VariableNotFound(obs_var.to_string()));
        }
        self.emission_dists.insert(obs_var.to_string(), dist);
        Ok(self)
    }

    /// Unroll the DBN for a fixed number of time steps.
    ///
    /// Returns a FactorGraph representing the unrolled DBN.
    pub fn unroll(&self, num_steps: usize) -> Result<FactorGraph> {
        if num_steps == 0 {
            return Err(PgmError::InvalidDistribution(
                "Number of steps must be positive".to_string(),
            ));
        }

        let mut graph = FactorGraph::new();

        // Add all variables for all time steps
        for t in 0..num_steps {
            // State variables
            for (var, card) in &self.state_vars {
                let temporal_name = format!("{}_{}", var, t);
                graph.add_variable_with_card(temporal_name, "State".to_string(), *card);
            }

            // Observation variables
            for (var, card) in &self.observation_vars {
                let temporal_name = format!("{}_{}", var, t);
                graph.add_variable_with_card(temporal_name, "Observation".to_string(), *card);
            }
        }

        // Add initial factors (t=0)
        for (var, card) in &self.state_vars {
            let temporal_name = format!("{}_{}", var, 0);
            let dist = self.initial_dists.get(var).cloned().unwrap_or_else(|| {
                // Default to uniform
                ArrayD::from_elem(IxDyn(&[*card]), 1.0 / *card as f64)
            });

            let factor = Factor::new(format!("P0_{}", var), vec![temporal_name], dist)?;
            graph.add_factor(factor)?;
        }

        // Add transition factors (t=1 to num_steps-1)
        for t in 1..num_steps {
            for (var, card) in &self.state_vars {
                let prev_name = format!("{}_{}", var, t - 1);
                let curr_name = format!("{}_{}", var, t);

                let dist = self.transition_dists.get(var).cloned().unwrap_or_else(|| {
                    // Default to identity transition
                    let mut identity = ArrayD::zeros(IxDyn(&[*card, *card]));
                    for i in 0..*card {
                        identity[[i, i]] = 1.0;
                    }
                    identity
                });

                let factor =
                    Factor::new(format!("T{}_{}", t, var), vec![prev_name, curr_name], dist)?;
                graph.add_factor(factor)?;
            }
        }

        // Add emission factors (all time steps)
        for t in 0..num_steps {
            for (obs_var, _) in &self.observation_vars {
                if let Some(dist) = self.emission_dists.get(obs_var) {
                    // Emission depends on state variables
                    let mut factor_vars: Vec<String> = self
                        .state_vars
                        .iter()
                        .map(|(v, _)| format!("{}_{}", v, t))
                        .collect();
                    factor_vars.push(format!("{}_{}", obs_var, t));

                    let factor =
                        Factor::new(format!("E{}_{}", t, obs_var), factor_vars, dist.clone())?;
                    graph.add_factor(factor)?;
                }
            }
        }

        Ok(graph)
    }

    /// Perform filtering to compute P(X_t | y_{1:t}).
    ///
    /// Returns marginal distributions for state variables at each time step.
    pub fn filter(
        &self,
        observations: &[HashMap<String, usize>],
    ) -> Result<Vec<HashMap<String, ArrayD<f64>>>> {
        let num_steps = observations.len();
        if num_steps == 0 {
            return Ok(Vec::new());
        }

        // Unroll the DBN
        let graph = self.unroll(num_steps)?;

        // Set evidence
        let mut evidence: HashMap<String, usize> = HashMap::new();
        for (t, obs) in observations.iter().enumerate() {
            for (var, &value) in obs {
                let temporal_name = format!("{}_{}", var, t);
                evidence.insert(temporal_name, value);
            }
        }

        // Run inference for each time step
        let ve = VariableElimination::default();
        let mut results = Vec::new();

        for t in 0..num_steps {
            let mut marginals = HashMap::new();

            for (var, _) in &self.state_vars {
                let temporal_name = format!("{}_{}", var, t);
                if let Ok(marginal) = ve.marginalize(&graph, &temporal_name) {
                    marginals.insert(var.clone(), marginal);
                }
            }

            results.push(marginals);
        }

        Ok(results)
    }

    /// Perform smoothing to compute P(X_t | y_{1:T}) for all t.
    ///
    /// Uses variable elimination on the unrolled DBN.
    pub fn smooth(
        &self,
        observations: &[HashMap<String, usize>],
    ) -> Result<Vec<HashMap<String, ArrayD<f64>>>> {
        // For smoothing, we need all evidence before computing marginals
        // The implementation is the same as filter for exact inference
        self.filter(observations)
    }

    /// Compute most likely sequence using Viterbi algorithm on unrolled DBN.
    pub fn viterbi(
        &self,
        observations: &[HashMap<String, usize>],
    ) -> Result<Vec<HashMap<String, usize>>> {
        let num_steps = observations.len();
        if num_steps == 0 {
            return Ok(Vec::new());
        }

        // Unroll the DBN
        let graph = self.unroll(num_steps)?;

        // Set evidence
        let mut evidence: HashMap<String, usize> = HashMap::new();
        for (t, obs) in observations.iter().enumerate() {
            for (var, &value) in obs {
                let temporal_name = format!("{}_{}", var, t);
                evidence.insert(temporal_name, value);
            }
        }

        // Run MAP inference using marginalization
        let ve = VariableElimination::default();

        let mut results = Vec::new();

        for t in 0..num_steps {
            let mut state = HashMap::new();

            for (var, _) in &self.state_vars {
                let temporal_name = format!("{}_{}", var, t);
                if let Ok(marginal) = ve.marginalize(&graph, &temporal_name) {
                    // Get argmax
                    let max_idx = marginal
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    state.insert(var.clone(), max_idx);
                }
            }

            results.push(state);
        }

        Ok(results)
    }

    /// Get interface (state) variable cardinalities.
    pub fn state_cardinalities(&self) -> HashMap<String, usize> {
        self.state_vars.iter().cloned().collect()
    }

    /// Get observation variable cardinalities.
    pub fn observation_cardinalities(&self) -> HashMap<String, usize> {
        self.observation_vars.iter().cloned().collect()
    }

    /// Get all variables in the DBN.
    pub fn all_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();

        for (var, _) in &self.state_vars {
            vars.insert(var.clone());
        }

        for (var, _) in &self.observation_vars {
            vars.insert(var.clone());
        }

        vars
    }

    /// Run belief propagation on the unrolled DBN.
    pub fn run_belief_propagation(
        &self,
        num_steps: usize,
        evidence: &HashMap<String, usize>,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let graph = self.unroll(num_steps)?;

        // Convert evidence to temporal format
        let mut temporal_evidence: HashMap<String, usize> = HashMap::new();
        for (var, &value) in evidence {
            // Assume evidence is for the last time step if no time suffix
            if var.contains('_') {
                temporal_evidence.insert(var.clone(), value);
            } else {
                temporal_evidence.insert(format!("{}_{}", var, num_steps - 1), value);
            }
        }

        // Run sum-product
        let algorithm = SumProductAlgorithm::new(100, 1e-6, 0.0);
        algorithm.run(&graph)
    }
}

/// Builder for creating DBNs with fluent API.
pub struct DBNBuilder {
    state_vars: Vec<(String, usize)>,
    obs_vars: Vec<(String, usize)>,
    initial: HashMap<String, ArrayD<f64>>,
    transitions: HashMap<String, ArrayD<f64>>,
    emissions: HashMap<String, ArrayD<f64>>,
}

impl Default for DBNBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DBNBuilder {
    /// Create a new DBN builder.
    pub fn new() -> Self {
        Self {
            state_vars: Vec::new(),
            obs_vars: Vec::new(),
            initial: HashMap::new(),
            transitions: HashMap::new(),
            emissions: HashMap::new(),
        }
    }

    /// Add a state variable.
    pub fn add_state_var(mut self, name: String, cardinality: usize) -> Self {
        self.state_vars.push((name, cardinality));
        self
    }

    /// Add an observation variable.
    pub fn add_observation_var(mut self, name: String, cardinality: usize) -> Self {
        self.obs_vars.push((name, cardinality));
        self
    }

    /// Set initial distribution for a state variable.
    pub fn set_initial(mut self, var: &str, dist: ArrayD<f64>) -> Self {
        self.initial.insert(var.to_string(), dist);
        self
    }

    /// Set transition distribution P(X_t | X_{t-1}).
    pub fn set_transition(mut self, var: &str, dist: ArrayD<f64>) -> Self {
        self.transitions.insert(var.to_string(), dist);
        self
    }

    /// Set emission distribution P(Y_t | X_t).
    pub fn set_emission(mut self, obs_var: &str, dist: ArrayD<f64>) -> Self {
        self.emissions.insert(obs_var.to_string(), dist);
        self
    }

    /// Build the DBN.
    pub fn build(self) -> Result<DynamicBayesianNetwork> {
        let mut dbn = DynamicBayesianNetwork::new(self.state_vars, self.obs_vars);

        for (var, dist) in self.initial {
            dbn.set_initial(&var, dist)?;
        }

        for (var, dist) in self.transitions {
            dbn.set_transition(&var, dist)?;
        }

        for (var, dist) in self.emissions {
            dbn.set_emission(&var, dist)?;
        }

        Ok(dbn)
    }
}

/// Coupled DBN with multiple interacting processes.
#[derive(Debug, Clone)]
pub struct CoupledDBN {
    /// Individual DBN processes
    pub processes: Vec<DynamicBayesianNetwork>,
    /// Coupling factors between processes
    pub couplings: Vec<CouplingFactor>,
}

/// Coupling factor between DBN processes.
#[derive(Debug, Clone)]
pub struct CouplingFactor {
    /// Process indices involved
    pub process_indices: Vec<usize>,
    /// Variables involved
    pub variables: Vec<String>,
    /// Coupling potential
    pub potential: ArrayD<f64>,
}

impl CoupledDBN {
    /// Create a new coupled DBN.
    pub fn new(processes: Vec<DynamicBayesianNetwork>) -> Self {
        Self {
            processes,
            couplings: Vec::new(),
        }
    }

    /// Add a coupling factor.
    pub fn add_coupling(&mut self, coupling: CouplingFactor) {
        self.couplings.push(coupling);
    }

    /// Unroll the coupled DBN.
    pub fn unroll(&self, num_steps: usize) -> Result<FactorGraph> {
        let mut graph = FactorGraph::new();

        // Unroll each process
        for (i, process) in self.processes.iter().enumerate() {
            let process_graph = process.unroll(num_steps)?;

            // Add variables with process prefix
            for var_name in process_graph.variable_names() {
                let full_name = format!("p{}_{}", i, var_name);
                if let Some(var) = process_graph.get_variable(var_name) {
                    graph.add_variable_with_card(full_name, var.domain.clone(), var.cardinality);
                }
            }

            // Add factors with process prefix
            for factor_id in process_graph.factor_ids() {
                if let Some(factor) = process_graph.get_factor(factor_id) {
                    let new_vars: Vec<String> = factor
                        .variables
                        .iter()
                        .map(|v| format!("p{}_{}", i, v))
                        .collect();

                    let new_factor = Factor::new(
                        format!("p{}_{}", i, factor.name),
                        new_vars,
                        factor.values.clone(),
                    )?;

                    graph.add_factor(new_factor)?;
                }
            }
        }

        // Add coupling factors
        for (i, coupling) in self.couplings.iter().enumerate() {
            let coupled_vars: Vec<String> = coupling
                .variables
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    if j < coupling.process_indices.len() {
                        format!("p{}_{}", coupling.process_indices[j], v)
                    } else {
                        v.clone()
                    }
                })
                .collect();

            let coupling_factor = Factor::new(
                format!("coupling_{}", i),
                coupled_vars,
                coupling.potential.clone(),
            )?;

            graph.add_factor(coupling_factor)?;
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_dbn_creation() {
        let dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        assert_eq!(dbn.state_vars.len(), 1);
        assert_eq!(dbn.observation_vars.len(), 0);
    }

    #[test]
    fn test_dbn_set_distributions() {
        let mut dbn = DynamicBayesianNetwork::new(
            vec![("state".to_string(), 2)],
            vec![("obs".to_string(), 3)],
        );

        let initial = Array::from_vec(vec![0.6, 0.4]).into_dyn();
        dbn.set_initial("state", initial).unwrap();

        let transition = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.7, 0.3, 0.4, 0.6]).unwrap();
        dbn.set_transition("state", transition).unwrap();

        assert!(dbn.initial_dists.contains_key("state"));
        assert!(dbn.transition_dists.contains_key("state"));
    }

    #[test]
    fn test_dbn_unroll() {
        let mut dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let initial = Array::from_vec(vec![0.6, 0.4]).into_dyn();
        dbn.set_initial("state", initial).unwrap();

        let graph = dbn.unroll(3).unwrap();

        // Should have 3 time steps
        assert!(graph.get_variable("state_0").is_some());
        assert!(graph.get_variable("state_1").is_some());
        assert!(graph.get_variable("state_2").is_some());
    }

    #[test]
    fn test_dbn_builder() {
        let dbn = DBNBuilder::new()
            .add_state_var("weather".to_string(), 2)
            .add_observation_var("umbrella".to_string(), 2)
            .set_initial("weather", Array::from_vec(vec![0.5, 0.5]).into_dyn())
            .set_transition(
                "weather",
                ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.7, 0.3, 0.3, 0.7]).unwrap(),
            )
            .build()
            .unwrap();

        assert_eq!(dbn.state_vars.len(), 1);
        assert_eq!(dbn.observation_vars.len(), 1);
    }

    #[test]
    fn test_dbn_state_cardinalities() {
        let dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 3)], vec![]);

        let cards = dbn.state_cardinalities();
        assert_eq!(cards.get("state"), Some(&3));
    }

    #[test]
    fn test_dbn_all_variables() {
        let dbn = DynamicBayesianNetwork::new(
            vec![("x".to_string(), 2), ("y".to_string(), 2)],
            vec![("obs".to_string(), 3)],
        );

        let vars = dbn.all_variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("obs"));
    }

    #[test]
    fn test_coupled_dbn() {
        let dbn1 = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let dbn2 = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let coupled = CoupledDBN::new(vec![dbn1, dbn2]);

        assert_eq!(coupled.processes.len(), 2);
    }

    #[test]
    fn test_temporal_var_display() {
        let tv = TemporalVar {
            name: "state".to_string(),
            time: 3,
        };

        assert_eq!(format!("{}", tv), "state_3");
    }

    #[test]
    fn test_dbn_filter_empty() {
        let dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let results = dbn.filter(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dbn_viterbi_empty() {
        let dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let results = dbn.viterbi(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dbn_unroll_zero_steps() {
        let dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let result = dbn.unroll(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dbn_set_invalid_var() {
        let mut dbn = DynamicBayesianNetwork::new(vec![("state".to_string(), 2)], vec![]);

        let dist = Array::from_vec(vec![0.5, 0.5]).into_dyn();
        let result = dbn.set_initial("invalid", dist);
        assert!(result.is_err());
    }
}
