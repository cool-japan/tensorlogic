//! Specialized model builders for common PGM types.
//!
//! This module provides convenient APIs for constructing and working with
//! common probabilistic graphical models:
//! - Bayesian Networks
//! - Hidden Markov Models (HMMs)
//! - Conditional Random Fields (CRFs)
//! - Markov Random Fields (MRFs)

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;

/// Bayesian Network builder.
///
/// Provides a convenient API for constructing directed acyclic graphical models
/// with conditional probability distributions.
pub struct BayesianNetwork {
    graph: FactorGraph,
    structure: HashMap<String, Vec<String>>, // var -> parents
}

impl BayesianNetwork {
    /// Create a new Bayesian Network.
    pub fn new() -> Self {
        Self {
            graph: FactorGraph::new(),
            structure: HashMap::new(),
        }
    }

    /// Add a variable node to the network.
    pub fn add_variable(&mut self, name: String, cardinality: usize) -> &mut Self {
        self.graph
            .add_variable_with_card(name.clone(), "Discrete".to_string(), cardinality);
        self.structure.insert(name, Vec::new());
        self
    }

    /// Add a conditional probability distribution P(child | parents).
    ///
    /// # Arguments
    /// * `child` - The dependent variable
    /// * `parents` - Parent variables that child depends on
    /// * `cpd` - Conditional probability table (dimensions: [parent1_card, ..., child_card])
    pub fn add_cpd(
        &mut self,
        child: String,
        parents: Vec<String>,
        cpd: ArrayD<f64>,
    ) -> Result<&mut Self> {
        // Verify child exists
        if self.graph.get_variable(&child).is_none() {
            return Err(PgmError::VariableNotFound(child));
        }

        // Verify parents exist
        for parent in &parents {
            if self.graph.get_variable(parent).is_none() {
                return Err(PgmError::VariableNotFound(parent.clone()));
            }
        }

        // Record structure
        self.structure.insert(child.clone(), parents.clone());

        // Create factor: variables = [parents..., child]
        let mut factor_vars = parents.clone();
        factor_vars.push(child.clone());

        let factor = Factor::new(format!("P({}|{:?})", child, parents), factor_vars, cpd)?;

        self.graph.add_factor(factor)?;
        Ok(self)
    }

    /// Add a prior probability P(variable) for a root node.
    pub fn add_prior(&mut self, variable: String, prior: ArrayD<f64>) -> Result<&mut Self> {
        let factor = Factor::new(format!("P({})", variable), vec![variable.clone()], prior)?;
        self.graph.add_factor(factor)?;
        self.structure.insert(variable, Vec::new());
        Ok(self)
    }

    /// Get the underlying factor graph.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }

    /// Check if the network is acyclic (DAG property).
    pub fn is_acyclic(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();

        for node in self.structure.keys() {
            if !visited.contains_key(node) && self.has_cycle(node, &mut visited, &mut rec_stack) {
                return false;
            }
        }

        true
    }

    fn has_cycle(
        &self,
        node: &str,
        visited: &mut HashMap<String, bool>,
        rec_stack: &mut HashMap<String, bool>,
    ) -> bool {
        visited.insert(node.to_string(), true);
        rec_stack.insert(node.to_string(), true);

        if let Some(parents) = self.structure.get(node) {
            for parent in parents {
                if !visited.contains_key(parent) {
                    if self.has_cycle(parent, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.get(parent) == Some(&true) {
                    return true;
                }
            }
        }

        rec_stack.insert(node.to_string(), false);
        false
    }

    /// Get topological ordering of variables (ancestors before descendants).
    pub fn topological_order(&self) -> Result<Vec<String>> {
        if !self.is_acyclic() {
            return Err(PgmError::InvalidGraph(
                "Network contains cycles".to_string(),
            ));
        }

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut children: HashMap<String, Vec<String>> = HashMap::new();

        // Build reverse graph (child -> parents becomes parent -> children)
        for (child, parents) in &self.structure {
            in_degree.insert(child.clone(), parents.len());
            for parent in parents {
                children
                    .entry(parent.clone())
                    .or_default()
                    .push(child.clone());
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(v, _)| v.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node.clone());

            if let Some(child_nodes) = children.get(&node) {
                for child in child_nodes {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(child.clone());
                        }
                    }
                }
            }
        }

        if result.len() != self.structure.len() {
            return Err(PgmError::InvalidGraph(
                "Could not compute topological order".to_string(),
            ));
        }

        Ok(result)
    }
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Hidden Markov Model builder.
///
/// A temporal model with hidden states and observations.
pub struct HiddenMarkovModel {
    graph: FactorGraph,
    #[allow(dead_code)]
    num_states: usize,
    #[allow(dead_code)]
    num_observations: usize,
    time_steps: usize,
}

impl HiddenMarkovModel {
    /// Create a new HMM.
    ///
    /// # Arguments
    /// * `num_states` - Number of hidden states
    /// * `num_observations` - Number of observable symbols
    /// * `time_steps` - Length of sequence
    pub fn new(num_states: usize, num_observations: usize, time_steps: usize) -> Self {
        let mut graph = FactorGraph::new();

        // Add hidden state variables
        for t in 0..time_steps {
            graph.add_variable_with_card(
                format!("state_{}", t),
                "HiddenState".to_string(),
                num_states,
            );
        }

        // Add observation variables
        for t in 0..time_steps {
            graph.add_variable_with_card(
                format!("obs_{}", t),
                "Observation".to_string(),
                num_observations,
            );
        }

        Self {
            graph,
            num_states,
            num_observations,
            time_steps,
        }
    }

    /// Set initial state distribution P(state_0).
    pub fn set_initial_distribution(&mut self, initial: ArrayD<f64>) -> Result<&mut Self> {
        let factor = Factor::new(
            "P(state_0)".to_string(),
            vec!["state_0".to_string()],
            initial,
        )?;
        self.graph.add_factor(factor)?;
        Ok(self)
    }

    /// Set transition matrix P(state_t | state_{t-1}).
    ///
    /// # Arguments
    /// * `transition` - Transition probabilities [from_state, to_state]
    pub fn set_transition_matrix(&mut self, transition: ArrayD<f64>) -> Result<&mut Self> {
        // Add transition factors for all time steps
        for t in 1..self.time_steps {
            let factor = Factor::new(
                format!("P(state_{}|state_{})", t, t - 1),
                vec![format!("state_{}", t - 1), format!("state_{}", t)],
                transition.clone(),
            )?;
            self.graph.add_factor(factor)?;
        }
        Ok(self)
    }

    /// Set emission matrix P(obs_t | state_t).
    ///
    /// # Arguments
    /// * `emission` - Emission probabilities [state, observation]
    pub fn set_emission_matrix(&mut self, emission: ArrayD<f64>) -> Result<&mut Self> {
        // Add emission factors for all time steps
        for t in 0..self.time_steps {
            let factor = Factor::new(
                format!("P(obs_{}|state_{})", t, t),
                vec![format!("state_{}", t), format!("obs_{}", t)],
                emission.clone(),
            )?;
            self.graph.add_factor(factor)?;
        }
        Ok(self)
    }

    /// Get the underlying factor graph.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }

    /// Perform filtering: compute P(state_t | obs_0:t).
    ///
    /// Uses variable elimination to compute the marginal distribution over
    /// the hidden state at time t given observations from 0 to t.
    pub fn filter(&self, observations: &[usize], t: usize) -> Result<ArrayD<f64>> {
        if t >= self.time_steps {
            return Err(PgmError::InvalidDistribution(format!(
                "Time step {} exceeds sequence length {}",
                t, self.time_steps
            )));
        }

        if t >= observations.len() {
            return Err(PgmError::InvalidDistribution(format!(
                "Not enough observations: need {} but got {}",
                t + 1,
                observations.len()
            )));
        }

        // Create a copy of the graph with evidence
        let mut evidence_graph = self.graph.clone();

        // Apply observations up to time t
        for (time, &obs_value) in observations.iter().enumerate().take(t + 1) {
            let obs_var = format!("obs_{}", time);

            // Add evidence factor: indicator function for observed value
            let mut evidence_values = vec![0.0; self.num_observations];
            evidence_values[obs_value] = 1.0;
            let evidence_factor = Factor::new(
                format!("evidence_{}", time),
                vec![obs_var.clone()],
                ArrayD::from_shape_vec(vec![self.num_observations], evidence_values)?,
            )?;
            evidence_graph.add_factor(evidence_factor)?;
        }

        // Use variable elimination to compute marginal
        use crate::variable_elimination::VariableElimination;
        let ve = VariableElimination::new();
        let state_var = format!("state_{}", t);
        ve.marginalize(&evidence_graph, &state_var)
    }

    /// Perform smoothing: compute P(state_t | obs_0:T).
    ///
    /// Uses variable elimination with all observations to compute the
    /// marginal distribution over the hidden state at time t.
    pub fn smooth(&self, observations: &[usize], t: usize) -> Result<ArrayD<f64>> {
        if t >= self.time_steps {
            return Err(PgmError::InvalidDistribution(format!(
                "Time step {} exceeds sequence length {}",
                t, self.time_steps
            )));
        }

        if observations.len() != self.time_steps {
            return Err(PgmError::InvalidDistribution(format!(
                "Expected {} observations but got {}",
                self.time_steps,
                observations.len()
            )));
        }

        // Create a copy of the graph with all evidence
        let mut evidence_graph = self.graph.clone();

        // Apply all observations
        for (time, &obs_value) in observations.iter().enumerate().take(self.time_steps) {
            let obs_var = format!("obs_{}", time);

            // Add evidence factor
            let mut evidence_values = vec![0.0; self.num_observations];
            evidence_values[obs_value] = 1.0;
            let evidence_factor = Factor::new(
                format!("evidence_{}", time),
                vec![obs_var.clone()],
                ArrayD::from_shape_vec(vec![self.num_observations], evidence_values)?,
            )?;
            evidence_graph.add_factor(evidence_factor)?;
        }

        // Use variable elimination to compute marginal
        use crate::variable_elimination::VariableElimination;
        let ve = VariableElimination::new();
        let state_var = format!("state_{}", t);
        ve.marginalize(&evidence_graph, &state_var)
    }

    /// Compute most likely state sequence (Viterbi algorithm).
    ///
    /// Finds the most probable sequence of hidden states given observations
    /// using dynamic programming.
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        if observations.len() != self.time_steps {
            return Err(PgmError::InvalidDistribution(format!(
                "Observations length {} does not match time steps {}",
                observations.len(),
                self.time_steps
            )));
        }

        // Create graph with evidence
        let mut evidence_graph = self.graph.clone();

        // Apply all observations
        for (time, &obs_value) in observations.iter().enumerate().take(self.time_steps) {
            let obs_var = format!("obs_{}", time);

            let mut evidence_values = vec![0.0; self.num_observations];
            evidence_values[obs_value] = 1.0;
            let evidence_factor = Factor::new(
                format!("evidence_{}", time),
                vec![obs_var.clone()],
                ArrayD::from_shape_vec(vec![self.num_observations], evidence_values)?,
            )?;
            evidence_graph.add_factor(evidence_factor)?;
        }

        // Use variable elimination with MAX to find MAP assignment
        use crate::variable_elimination::VariableElimination;
        let ve = VariableElimination::new();
        let assignment = ve.map(&evidence_graph)?;

        // Extract state sequence in temporal order
        let mut sequence = Vec::new();
        for t in 0..self.time_steps {
            let state_var = format!("state_{}", t);
            if let Some(&state) = assignment.get(&state_var) {
                sequence.push(state);
            } else {
                return Err(PgmError::VariableNotFound(state_var));
            }
        }

        Ok(sequence)
    }
}

/// Markov Random Field builder (undirected graphical model).
pub struct MarkovRandomField {
    graph: FactorGraph,
}

impl MarkovRandomField {
    /// Create a new MRF.
    pub fn new() -> Self {
        Self {
            graph: FactorGraph::new(),
        }
    }

    /// Add a variable node.
    pub fn add_variable(&mut self, name: String, cardinality: usize) -> &mut Self {
        self.graph
            .add_variable_with_card(name, "Discrete".to_string(), cardinality);
        self
    }

    /// Add a pairwise potential φ(x_i, x_j).
    pub fn add_pairwise_potential(
        &mut self,
        var1: String,
        var2: String,
        potential: ArrayD<f64>,
    ) -> Result<&mut Self> {
        let factor = Factor::new(
            format!("φ({},{})", var1, var2),
            vec![var1.clone(), var2.clone()],
            potential,
        )?;
        self.graph.add_factor(factor)?;
        Ok(self)
    }

    /// Add a unary potential φ(x_i).
    pub fn add_unary_potential(
        &mut self,
        var: String,
        potential: ArrayD<f64>,
    ) -> Result<&mut Self> {
        let factor = Factor::new(format!("φ({})", var), vec![var.clone()], potential)?;
        self.graph.add_factor(factor)?;
        Ok(self)
    }

    /// Get the underlying factor graph.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }
}

impl Default for MarkovRandomField {
    fn default() -> Self {
        Self::new()
    }
}

/// Conditional Random Field builder (discriminative model for structured prediction).
pub struct ConditionalRandomField {
    graph: FactorGraph,
    input_vars: Vec<String>,
    output_vars: Vec<String>,
}

impl ConditionalRandomField {
    /// Create a new CRF.
    pub fn new() -> Self {
        Self {
            graph: FactorGraph::new(),
            input_vars: Vec::new(),
            output_vars: Vec::new(),
        }
    }

    /// Add an input (observed) variable.
    pub fn add_input_variable(&mut self, name: String, cardinality: usize) -> &mut Self {
        self.graph
            .add_variable_with_card(name.clone(), "Input".to_string(), cardinality);
        self.input_vars.push(name);
        self
    }

    /// Add an output (label) variable.
    pub fn add_output_variable(&mut self, name: String, cardinality: usize) -> &mut Self {
        self.graph
            .add_variable_with_card(name.clone(), "Output".to_string(), cardinality);
        self.output_vars.push(name);
        self
    }

    /// Add a feature function (factor).
    pub fn add_feature(
        &mut self,
        name: String,
        variables: Vec<String>,
        potential: ArrayD<f64>,
    ) -> Result<&mut Self> {
        let factor = Factor::new(format!("feature_{}", name), variables, potential)?;
        self.graph.add_factor(factor)?;
        Ok(self)
    }

    /// Get the underlying factor graph.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }
}

impl Default for ConditionalRandomField {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_bayesian_network_creation() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("x".to_string(), 2);
        bn.add_variable("y".to_string(), 2);

        assert!(bn.graph().get_variable("x").is_some());
        assert!(bn.graph().get_variable("y").is_some());
    }

    #[test]
    fn test_bayesian_network_cpd() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("x".to_string(), 2);
        bn.add_variable("y".to_string(), 2);

        let prior = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        bn.add_prior("x".to_string(), prior).unwrap();

        let cpd = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        bn.add_cpd("y".to_string(), vec!["x".to_string()], cpd)
            .unwrap();

        assert_eq!(bn.graph().num_factors(), 2);
    }

    #[test]
    fn test_bayesian_network_acyclic() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("x".to_string(), 2);
        bn.add_variable("y".to_string(), 2);

        let cpd = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        bn.add_cpd("y".to_string(), vec!["x".to_string()], cpd)
            .unwrap();

        assert!(bn.is_acyclic());
    }

    #[test]
    fn test_bayesian_network_topological_order() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("x".to_string(), 2);
        bn.add_variable("y".to_string(), 2);
        bn.add_variable("z".to_string(), 2);

        let cpd_y = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        bn.add_cpd("y".to_string(), vec!["x".to_string()], cpd_y)
            .unwrap();

        let cpd_z = Array::from_shape_vec(vec![2, 2], vec![0.8, 0.2, 0.3, 0.7])
            .unwrap()
            .into_dyn();
        bn.add_cpd("z".to_string(), vec!["y".to_string()], cpd_z)
            .unwrap();

        let order = bn.topological_order().unwrap();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_hmm_creation() {
        let hmm = HiddenMarkovModel::new(3, 2, 5);
        assert_eq!(hmm.graph().num_variables(), 10); // 5 states + 5 observations
    }

    #[test]
    fn test_hmm_parameters() {
        let mut hmm = HiddenMarkovModel::new(2, 2, 3);

        let initial = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        hmm.set_initial_distribution(initial).unwrap();

        let transition = Array::from_shape_vec(vec![2, 2], vec![0.7, 0.3, 0.4, 0.6])
            .unwrap()
            .into_dyn();
        hmm.set_transition_matrix(transition).unwrap();

        let emission = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        hmm.set_emission_matrix(emission).unwrap();

        assert!(hmm.graph().num_factors() > 0);
    }

    #[test]
    fn test_hmm_filtering() {
        let mut hmm = HiddenMarkovModel::new(2, 2, 3);

        let initial = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        hmm.set_initial_distribution(initial).unwrap();

        let transition = Array::from_shape_vec(vec![2, 2], vec![0.7, 0.3, 0.4, 0.6])
            .unwrap()
            .into_dyn();
        hmm.set_transition_matrix(transition).unwrap();

        let emission = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        hmm.set_emission_matrix(emission).unwrap();

        // Filter with observations
        let observations = vec![0, 1, 0];
        let result = hmm.filter(&observations, 1);
        assert!(result.is_ok());

        let marginal = result.unwrap();
        assert_eq!(marginal.len(), 2);
        // Should be normalized
        let sum: f64 = marginal.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hmm_smoothing() {
        let mut hmm = HiddenMarkovModel::new(2, 2, 3);

        let initial = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        hmm.set_initial_distribution(initial).unwrap();

        let transition = Array::from_shape_vec(vec![2, 2], vec![0.7, 0.3, 0.4, 0.6])
            .unwrap()
            .into_dyn();
        hmm.set_transition_matrix(transition).unwrap();

        let emission = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        hmm.set_emission_matrix(emission).unwrap();

        // Smooth with all observations
        let observations = vec![0, 1, 0];
        let result = hmm.smooth(&observations, 1);
        assert!(result.is_ok());

        let marginal = result.unwrap();
        assert_eq!(marginal.len(), 2);
        let sum: f64 = marginal.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hmm_viterbi() {
        let mut hmm = HiddenMarkovModel::new(2, 2, 3);

        let initial = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        hmm.set_initial_distribution(initial).unwrap();

        let transition = Array::from_shape_vec(vec![2, 2], vec![0.7, 0.3, 0.4, 0.6])
            .unwrap()
            .into_dyn();
        hmm.set_transition_matrix(transition).unwrap();

        let emission = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        hmm.set_emission_matrix(emission).unwrap();

        // Run Viterbi
        let observations = vec![0, 1, 0];
        let result = hmm.viterbi(&observations);
        assert!(result.is_ok());

        let sequence = result.unwrap();
        assert_eq!(sequence.len(), 3);
        // Each state should be valid (0 or 1)
        for state in sequence {
            assert!(state < 2);
        }
    }

    #[test]
    fn test_mrf_creation() {
        let mut mrf = MarkovRandomField::new();
        mrf.add_variable("x".to_string(), 2);
        mrf.add_variable("y".to_string(), 2);

        let potential = Array::from_shape_vec(vec![2, 2], vec![1.0, 0.5, 0.5, 1.0])
            .unwrap()
            .into_dyn();
        mrf.add_pairwise_potential("x".to_string(), "y".to_string(), potential)
            .unwrap();

        assert_eq!(mrf.graph().num_factors(), 1);
    }

    #[test]
    fn test_crf_creation() {
        let mut crf = ConditionalRandomField::new();
        crf.add_input_variable("x".to_string(), 3);
        crf.add_output_variable("y".to_string(), 2);

        let feature = Array::from_shape_vec(vec![3, 2], vec![1.0, 0.5, 0.8, 0.2, 0.6, 0.4])
            .unwrap()
            .into_dyn();
        crf.add_feature(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            feature,
        )
        .unwrap();

        assert_eq!(crf.graph().num_factors(), 1);
    }
}
