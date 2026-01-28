//! QuantRS2 integration hooks for probabilistic graphical models.
//!
//! This module provides integration between tensorlogic-quantrs-hooks and the QuantRS2
//! probabilistic programming ecosystem. It defines traits and utilities for seamless
//! interoperability between PGM inference and QuantRS2 distributions and models.
//!
//! # Architecture
//!
//! ```text
//! TensorLogic PGM ←→ QuantRS2 Distributions
//!       ↓                      ↓
//!   FactorGraph ←→ Probabilistic Models
//!       ↓                      ↓
//!   Inference   ←→  Sampling/Optimization
//! ```
//!
//! # Integration Points
//!
//! 1. **Distribution Conversion**: Factor ↔ QuantRS Distribution
//! 2. **Model Export**: FactorGraph → QuantRS ProbabilisticModel
//! 3. **Inference Queries**: Unified query interface
//! 4. **Parameter Learning**: Hook into QuantRS optimizers
//! 5. **Sampling**: Bridge to QuantRS MCMC samplers

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;
use scirs2_core::ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for converting between PGM factors and QuantRS distributions.
///
/// This enables seamless integration with QuantRS2's probabilistic modeling framework.
pub trait QuantRSDistribution {
    /// Convert a factor to a QuantRS-compatible distribution.
    ///
    /// # Returns
    ///
    /// A normalized probability distribution that can be used with QuantRS2 samplers
    /// and inference algorithms.
    fn to_quantrs_distribution(&self) -> Result<DistributionExport>;

    /// Create a factor from a QuantRS distribution.
    ///
    /// # Arguments
    ///
    /// * `dist` - The QuantRS distribution to convert
    ///
    /// # Returns
    ///
    /// A Factor representation suitable for PGM inference.
    fn from_quantrs_distribution(dist: &DistributionExport) -> Result<Self>
    where
        Self: Sized;

    /// Check if the distribution is normalized.
    fn is_normalized(&self) -> bool;

    /// Get the support (valid values) of the distribution.
    fn support(&self) -> Vec<Vec<usize>>;
}

/// Exported distribution format compatible with QuantRS2.
///
/// This structure can be serialized and used across the COOLJAPAN ecosystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionExport {
    /// Variable names
    pub variables: Vec<String>,
    /// Domain sizes (cardinalities) for each variable
    pub cardinalities: Vec<usize>,
    /// Probability values (flattened tensor)
    pub probabilities: Vec<f64>,
    /// Shape of the probability tensor
    pub shape: Vec<usize>,
    /// Metadata for integration
    pub metadata: DistributionMetadata,
}

/// Metadata for distribution export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionMetadata {
    /// Distribution type (e.g., "categorical", "gaussian", "conditional")
    pub distribution_type: String,
    /// Whether the distribution is normalized
    pub normalized: bool,
    /// Optional parameter names
    pub parameter_names: Vec<String>,
    /// Optional tags for categorization
    pub tags: Vec<String>,
}

impl QuantRSDistribution for Factor {
    fn to_quantrs_distribution(&self) -> Result<DistributionExport> {
        // Get cardinalities from shape
        let cardinalities: Vec<usize> = self.values.shape().to_vec();

        // Flatten values
        let probabilities: Vec<f64> = self.values.iter().copied().collect();

        // Check normalization
        let sum: f64 = probabilities.iter().sum();
        let normalized = (sum - 1.0).abs() < 1e-6;

        Ok(DistributionExport {
            variables: self.variables.clone(),
            cardinalities,
            probabilities,
            shape: self.values.shape().to_vec(),
            metadata: DistributionMetadata {
                distribution_type: "categorical".to_string(),
                normalized,
                parameter_names: vec![],
                tags: vec!["pgm".to_string(), "factor".to_string()],
            },
        })
    }

    fn from_quantrs_distribution(dist: &DistributionExport) -> Result<Self> {
        let array = ArrayD::from_shape_vec(dist.shape.clone(), dist.probabilities.clone())
            .map_err(|e| PgmError::InvalidGraph(format!("Array creation failed: {}", e)))?;

        Factor::new("quantrs_import".to_string(), dist.variables.clone(), array)
    }

    fn is_normalized(&self) -> bool {
        let sum: f64 = self.values.iter().sum();
        (sum - 1.0).abs() < 1e-6
    }

    fn support(&self) -> Vec<Vec<usize>> {
        let shape = self.values.shape();
        let mut support = Vec::new();

        fn generate_indices(shape: &[usize], current: Vec<usize>, support: &mut Vec<Vec<usize>>) {
            if current.len() == shape.len() {
                support.push(current);
                return;
            }

            let dim = current.len();
            for i in 0..shape[dim] {
                let mut next = current.clone();
                next.push(i);
                generate_indices(shape, next, support);
            }
        }

        generate_indices(shape, vec![], &mut support);
        support
    }
}

/// Trait for models that can export to QuantRS2 format.
pub trait QuantRSModelExport {
    /// Export the model to a QuantRS-compatible format.
    fn to_quantrs_model(&self) -> Result<ModelExport>;

    /// Get model statistics for QuantRS integration.
    fn model_stats(&self) -> ModelStatistics;
}

/// Exported model format compatible with QuantRS2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExport {
    /// Model type (e.g., "bayesian_network", "markov_random_field")
    pub model_type: String,
    /// Variable definitions
    pub variables: Vec<VariableDefinition>,
    /// Factor definitions
    pub factors: Vec<FactorDefinition>,
    /// Model structure (edges, dependencies)
    pub structure: ModelStructure,
    /// Metadata
    pub metadata: ModelMetadata,
}

/// Variable definition for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDefinition {
    /// Variable name
    pub name: String,
    /// Domain type
    pub domain: String,
    /// Cardinality (number of possible values)
    pub cardinality: usize,
    /// Optional domain values
    pub domain_values: Option<Vec<String>>,
}

/// Factor definition for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorDefinition {
    /// Factor name
    pub name: String,
    /// Scope (variables involved)
    pub scope: Vec<String>,
    /// Distribution export
    pub distribution: DistributionExport,
}

/// Model structure definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStructure {
    /// Type of structure ("directed", "undirected", "factor_graph")
    pub structure_type: String,
    /// Edges (for directed/undirected graphs)
    pub edges: Vec<(String, String)>,
    /// Cliques (for MRFs)
    pub cliques: Vec<Vec<String>>,
}

/// Model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Description
    pub description: String,
    /// Creation timestamp
    pub created_at: String,
    /// Tags
    pub tags: Vec<String>,
}

/// Model statistics for QuantRS integration.
#[derive(Debug, Clone)]
pub struct ModelStatistics {
    /// Number of variables
    pub num_variables: usize,
    /// Number of factors
    pub num_factors: usize,
    /// Average factor size
    pub avg_factor_size: f64,
    /// Maximum factor size
    pub max_factor_size: usize,
    /// Treewidth (if computed)
    pub treewidth: Option<usize>,
}

impl QuantRSModelExport for FactorGraph {
    fn to_quantrs_model(&self) -> Result<ModelExport> {
        // Export variables
        let variables: Vec<VariableDefinition> = self
            .variables()
            .map(|(name, var)| VariableDefinition {
                name: name.clone(),
                domain: var.domain.clone(),
                cardinality: var.cardinality,
                domain_values: None,
            })
            .collect();

        // Export factors
        let factors: Vec<FactorDefinition> = self
            .factors()
            .map(|factor| {
                Ok(FactorDefinition {
                    name: factor.name.clone(),
                    scope: factor.variables.clone(),
                    distribution: factor.to_quantrs_distribution()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build structure
        let edges = Vec::new();
        let mut cliques = Vec::new();

        for factor in self.factors() {
            if factor.variables.len() > 1 {
                cliques.push(factor.variables.clone());
            }
        }

        Ok(ModelExport {
            model_type: "factor_graph".to_string(),
            variables,
            factors,
            structure: ModelStructure {
                structure_type: "undirected".to_string(),
                edges,
                cliques,
            },
            metadata: ModelMetadata {
                name: "Exported FactorGraph".to_string(),
                description: "Factor graph exported from tensorlogic-quantrs-hooks".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                tags: vec!["pgm".to_string(), "factor_graph".to_string()],
            },
        })
    }

    fn model_stats(&self) -> ModelStatistics {
        let num_variables = self.num_variables();
        let num_factors = self.num_factors();

        let avg_factor_size = if num_factors > 0 {
            self.factors().map(|f| f.variables.len()).sum::<usize>() as f64 / num_factors as f64
        } else {
            0.0
        };

        let max_factor_size = self.factors().map(|f| f.variables.len()).max().unwrap_or(0);

        ModelStatistics {
            num_variables,
            num_factors,
            avg_factor_size,
            max_factor_size,
            treewidth: None,
        }
    }
}

/// Trait for probabilistic inference queries compatible with QuantRS2.
pub trait QuantRSInferenceQuery {
    /// Execute a marginal query and return QuantRS-compatible distribution.
    fn query_marginal_quantrs(&self, variable: &str) -> Result<DistributionExport>;

    /// Execute a conditional query.
    fn query_conditional_quantrs(
        &self,
        variable: &str,
        evidence: &HashMap<String, usize>,
    ) -> Result<DistributionExport>;

    /// Execute a MAP (maximum a posteriori) query.
    fn query_map_quantrs(&self) -> Result<HashMap<String, usize>>;
}

/// Parameter learning interface for QuantRS integration.
///
/// This trait enables parameter estimation using QuantRS2 optimization algorithms.
pub trait QuantRSParameterLearning {
    /// Learn parameters from data using maximum likelihood estimation.
    fn learn_parameters_ml(&mut self, data: &[QuantRSAssignment]) -> Result<()>;

    /// Learn parameters using Bayesian estimation with priors.
    fn learn_parameters_bayesian(
        &mut self,
        data: &[QuantRSAssignment],
        priors: &HashMap<String, ArrayD<f64>>,
    ) -> Result<()>;

    /// Get current parameters as QuantRS distributions.
    fn get_parameters(&self) -> Result<Vec<DistributionExport>>;

    /// Set parameters from QuantRS distributions.
    fn set_parameters(&mut self, params: &[DistributionExport]) -> Result<()>;
}

/// Assignment of values to variables (for learning and QuantRS integration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantRSAssignment {
    /// Variable assignments
    pub assignments: HashMap<String, usize>,
}

impl QuantRSAssignment {
    /// Create a new assignment.
    pub fn new(assignments: HashMap<String, usize>) -> Self {
        Self { assignments }
    }

    /// Get the value assigned to a variable.
    pub fn get(&self, variable: &str) -> Option<usize> {
        self.assignments.get(variable).copied()
    }

    /// Create from a simple HashMap (compatibility with sampling module).
    pub fn from_hashmap(assignments: HashMap<String, usize>) -> Self {
        Self { assignments }
    }

    /// Convert to a simple HashMap (compatibility with sampling module).
    pub fn to_hashmap(&self) -> HashMap<String, usize> {
        self.assignments.clone()
    }
}

/// Hook for MCMC sampling integration with QuantRS2.
pub trait QuantRSSamplingHook {
    /// Generate samples using QuantRS2-compatible sampler.
    fn sample_quantrs(&self, num_samples: usize) -> Result<Vec<QuantRSAssignment>>;

    /// Compute log-likelihood for QuantRS integration.
    fn log_likelihood(&self, assignment: &QuantRSAssignment) -> Result<f64>;

    /// Compute unnormalized probability (potential).
    fn unnormalized_probability(&self, assignment: &QuantRSAssignment) -> Result<f64>;
}

// ============================================================================
// Quantum Computing Integration Traits
// ============================================================================

/// Configuration for quantum annealing optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingConfig {
    /// Number of annealing steps
    pub num_steps: usize,
    /// Total annealing time
    pub annealing_time: f64,
    /// Number of samples per run
    pub num_samples: usize,
    /// Initial temperature (for simulated annealing)
    pub initial_temperature: f64,
    /// Final temperature (for simulated annealing)
    pub final_temperature: f64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            num_steps: 100,
            annealing_time: 10.0,
            num_samples: 100,
            initial_temperature: 10.0,
            final_temperature: 0.01,
        }
    }
}

impl AnnealingConfig {
    /// Create a new annealing configuration.
    pub fn new(num_steps: usize, annealing_time: f64) -> Self {
        Self {
            num_steps,
            annealing_time,
            ..Default::default()
        }
    }

    /// Set the number of samples.
    pub fn with_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }

    /// Set the temperature schedule.
    pub fn with_temperature(mut self, initial: f64, final_temp: f64) -> Self {
        self.initial_temperature = initial;
        self.final_temperature = final_temp;
        self
    }
}

/// Solution from quantum annealing or QAOA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSolution {
    /// Variable assignments
    pub assignments: HashMap<String, usize>,
    /// Objective value (energy)
    pub objective_value: f64,
    /// Solution quality indicator (lower is better)
    pub quality: f64,
    /// Number of iterations/shots used
    pub iterations: usize,
    /// Additional metadata
    pub metadata: QuantumSolutionMetadata,
}

/// Metadata for quantum solutions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSolutionMetadata {
    /// Algorithm used
    pub algorithm: String,
    /// Number of QAOA layers (if applicable)
    pub num_layers: Option<usize>,
    /// Optimal parameters found
    pub optimal_params: Option<Vec<f64>>,
    /// Time taken in seconds
    pub time_seconds: Option<f64>,
}

impl QuantumSolution {
    /// Create a new quantum solution.
    pub fn new(assignments: HashMap<String, usize>, objective_value: f64, algorithm: &str) -> Self {
        Self {
            assignments,
            objective_value,
            quality: objective_value.abs(),
            iterations: 1,
            metadata: QuantumSolutionMetadata {
                algorithm: algorithm.to_string(),
                num_layers: None,
                optimal_params: None,
                time_seconds: None,
            },
        }
    }

    /// Get variable assignment.
    pub fn get(&self, variable: &str) -> Option<usize> {
        self.assignments.get(variable).copied()
    }
}

/// Trait for quantum-enhanced inference on factor graphs.
///
/// This trait provides methods for using quantum algorithms (QAOA, quantum annealing)
/// to perform inference tasks on probabilistic graphical models.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_quantrs_hooks::{FactorGraph, QuantumInference};
/// use std::collections::HashMap;
///
/// let mut graph = FactorGraph::new();
/// graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
/// graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);
///
/// // Solve using QAOA
/// let solution = graph.solve_qaoa(2).unwrap();
/// println!("Best assignment: {:?}", solution);
/// ```
pub trait QuantumInference {
    /// Solve the optimization problem using QAOA (Quantum Approximate Optimization Algorithm).
    ///
    /// QAOA maps the factor graph to a quantum circuit and finds the optimal
    /// variable assignment that maximizes the joint probability (or minimizes energy).
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of QAOA layers (p parameter). More layers give
    ///   better approximation but require more quantum resources.
    ///
    /// # Returns
    ///
    /// A map from variable names to their optimal values.
    fn solve_qaoa(&self, num_layers: usize) -> Result<HashMap<String, usize>>;

    /// Compute marginal distributions using quantum sampling.
    ///
    /// This method uses quantum circuits to sample from the joint distribution
    /// and estimates marginal probabilities from the samples.
    ///
    /// # Arguments
    ///
    /// * `num_shots` - Number of measurement shots for sampling.
    ///
    /// # Returns
    ///
    /// A map from variable names to their marginal probability distributions.
    fn quantum_marginals(&self, num_shots: usize) -> Result<HashMap<String, ArrayD<f64>>>;

    /// Compute the partition function using quantum amplitude estimation.
    ///
    /// This is useful for computing normalized probabilities and
    /// free energy.
    fn quantum_partition_function(&self) -> Result<f64>;
}

/// Trait for quantum annealing optimization.
///
/// Quantum annealing is a metaheuristic that uses quantum fluctuations
/// to find the global minimum of an objective function.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_quantrs_hooks::{FactorGraph, QuantumAnnealing, AnnealingConfig};
/// use tensorlogic_quantrs_hooks::quantum_circuit::QUBOProblem;
///
/// let mut graph = FactorGraph::new();
/// graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
///
/// // Convert to QUBO
/// let qubo = graph.to_qubo().unwrap();
///
/// // Run annealing
/// let config = AnnealingConfig::default();
/// let solution = graph.anneal(&config).unwrap();
/// ```
pub trait QuantumAnnealing {
    /// Convert the factor graph to a QUBO (Quadratic Unconstrained Binary Optimization) problem.
    ///
    /// QUBO is the natural formulation for quantum annealing.
    fn to_qubo(&self) -> Result<crate::quantum_circuit::QUBOProblem>;

    /// Run quantum annealing to find the optimal assignment.
    ///
    /// # Arguments
    ///
    /// * `config` - Annealing configuration parameters.
    ///
    /// # Returns
    ///
    /// The optimal solution found by annealing.
    fn anneal(&self, config: &AnnealingConfig) -> Result<QuantumSolution>;

    /// Run multiple annealing runs and return the best solution.
    ///
    /// # Arguments
    ///
    /// * `config` - Annealing configuration parameters.
    /// * `num_runs` - Number of independent annealing runs.
    ///
    /// # Returns
    ///
    /// The best solution across all runs.
    fn anneal_multiple(&self, config: &AnnealingConfig, num_runs: usize)
        -> Result<QuantumSolution>;
}

// Implement QuantumInference for FactorGraph
impl QuantumInference for FactorGraph {
    fn solve_qaoa(&self, num_layers: usize) -> Result<HashMap<String, usize>> {
        use crate::quantum_circuit::{factor_graph_to_qubo, QAOAConfig};
        use crate::quantum_simulation::{run_qaoa, QuantumSimulationBackend};

        let qubo = factor_graph_to_qubo(self)?;
        let config = QAOAConfig::new(num_layers);
        let backend = QuantumSimulationBackend::new();
        let result = run_qaoa(&qubo, &config, &backend)?;

        // Convert result to HashMap
        let var_names: Vec<String> = self.variable_names().cloned().collect();
        let mut assignments: HashMap<String, usize> = HashMap::new();

        let solution: &Vec<usize> = &result.best_solution;
        for (idx, &value) in solution.iter().enumerate() {
            if idx < var_names.len() {
                let var_name: &String = &var_names[idx];
                assignments.insert(var_name.clone(), value);
            }
        }

        Ok(assignments)
    }

    fn quantum_marginals(&self, num_shots: usize) -> Result<HashMap<String, ArrayD<f64>>> {
        use crate::quantum_simulation::{QuantumSimulationBackend, SimulationConfig};

        // Create backend and run quantum sampling
        let config = SimulationConfig::with_shots(num_shots);
        let backend = QuantumSimulationBackend::with_config(config);
        let samples = backend.quantum_sample(self, num_shots)?;

        // Compute marginals from samples
        let mut counts: HashMap<String, Vec<usize>> = HashMap::new();
        let var_names: Vec<String> = self.variable_names().cloned().collect();

        for var in &var_names {
            counts.insert(var.clone(), vec![0, 0]); // Binary variables
        }

        for sample in &samples {
            for (var, &value) in sample {
                if let Some(count) = counts.get_mut(var) {
                    if value < count.len() {
                        count[value] += 1;
                    }
                }
            }
        }

        // Convert counts to probabilities
        let mut marginals: HashMap<String, ArrayD<f64>> = HashMap::new();
        let total = samples.len() as f64;

        for (var, count_vec) in counts {
            let probs: Vec<f64> = count_vec.iter().map(|&c| c as f64 / total).collect();
            let shape = vec![probs.len()];
            let arrd = ArrayD::from_shape_vec(shape, probs)
                .map_err(|e| PgmError::InvalidDistribution(format!("Reshape failed: {}", e)))?;
            marginals.insert(var, arrd);
        }

        Ok(marginals)
    }

    fn quantum_partition_function(&self) -> Result<f64> {
        // Simplified: sum over all configurations
        // In practice, would use quantum amplitude estimation
        let mut z = 0.0;
        let var_names: Vec<String> = self.variable_names().cloned().collect();
        let cardinalities: Vec<usize> = var_names
            .iter()
            .filter_map(|name| self.get_variable(name).map(|v| v.cardinality))
            .collect();

        let total_configs: usize = cardinalities.iter().product();

        for config_idx in 0..total_configs {
            let mut assignment = HashMap::new();
            let mut temp = config_idx;

            for (i, &card) in cardinalities.iter().enumerate().rev() {
                assignment.insert(var_names[i].clone(), temp % card);
                temp /= card;
            }

            // Compute unnormalized probability for this configuration
            let mut prob = 1.0;
            for factor in self.factors() {
                let mut indices = Vec::new();
                for var in &factor.variables {
                    if let Some(&val) = assignment.get(var) {
                        indices.push(val);
                    }
                }
                if !indices.is_empty() {
                    prob *= factor.values[indices.as_slice()];
                }
            }

            z += prob;
        }

        Ok(z)
    }
}

// Implement QuantumAnnealing for FactorGraph
impl QuantumAnnealing for FactorGraph {
    fn to_qubo(&self) -> Result<crate::quantum_circuit::QUBOProblem> {
        crate::quantum_circuit::factor_graph_to_qubo(self)
    }

    fn anneal(&self, config: &AnnealingConfig) -> Result<QuantumSolution> {
        // Use classical simulated annealing as placeholder
        // Full quantum annealing would require hardware integration
        use scirs2_core::random::thread_rng;

        let qubo = self.to_qubo()?;
        let num_vars = qubo.num_variables;
        let var_names: Vec<String> = self.variable_names().cloned().collect();

        // Initialize random solution using f64 and converting
        let mut rng = thread_rng();
        let mut best_solution: Vec<usize> = (0..num_vars)
            .map(|_| if rng.random::<f64>() < 0.5 { 0 } else { 1 })
            .collect();

        // Compute initial value
        let compute_value = |sol: &[usize]| -> f64 {
            let mut val = qubo.offset;
            for i in 0..num_vars {
                val += qubo.linear[i] * sol[i] as f64;
                for j in (i + 1)..num_vars {
                    val += qubo.quadratic[[i, j]] * (sol[i] * sol[j]) as f64;
                }
            }
            val
        };

        let mut best_value = compute_value(&best_solution);

        // Simulated annealing loop
        let mut current = best_solution.clone();
        let mut current_value = best_value;

        for step in 0..config.num_steps {
            let temp = config.annealing_time * (1.0 - step as f64 / config.num_steps as f64);

            // Flip a random bit using f64 random
            let flip_idx = (rng.random::<f64>() * num_vars as f64) as usize % num_vars;
            current[flip_idx] = 1 - current[flip_idx];

            let new_value = compute_value(&current);
            let delta = new_value - current_value;

            if delta < 0.0 || rng.random::<f64>() < (-delta / temp.max(1e-10)).exp() {
                current_value = new_value;
                if current_value < best_value {
                    best_value = current_value;
                    best_solution = current.clone();
                }
            } else {
                // Revert flip
                current[flip_idx] = 1 - current[flip_idx];
            }
        }

        // Convert solution to HashMap
        let mut assignments: HashMap<String, usize> = HashMap::new();
        for (idx, &val) in best_solution.iter().enumerate() {
            if idx < var_names.len() {
                let var_name: &String = &var_names[idx];
                assignments.insert(var_name.clone(), val);
            }
        }

        Ok(QuantumSolution {
            assignments,
            objective_value: best_value,
            quality: best_value.abs(),
            iterations: config.num_steps,
            metadata: QuantumSolutionMetadata {
                algorithm: "simulated_annealing".to_string(),
                num_layers: None,
                optimal_params: None,
                time_seconds: None,
            },
        })
    }

    fn anneal_multiple(
        &self,
        config: &AnnealingConfig,
        num_runs: usize,
    ) -> Result<QuantumSolution> {
        let mut best_solution: Option<QuantumSolution> = None;

        for _ in 0..num_runs {
            let solution = self.anneal(config)?;

            match &best_solution {
                None => best_solution = Some(solution),
                Some(best) => {
                    if solution.objective_value < best.objective_value {
                        best_solution = Some(solution);
                    }
                }
            }
        }

        best_solution.ok_or_else(|| PgmError::InvalidGraph("No solution found".to_string()))
    }
}

/// Utility functions for QuantRS integration.
pub mod utils {
    use super::*;

    /// Convert a factor graph to JSON for QuantRS export.
    pub fn export_to_json(graph: &FactorGraph) -> Result<String> {
        let model = graph.to_quantrs_model()?;
        serde_json::to_string_pretty(&model)
            .map_err(|e| PgmError::InvalidGraph(format!("JSON serialization failed: {}", e)))
    }

    /// Import a factor graph from JSON.
    pub fn import_from_json(json: &str) -> Result<ModelExport> {
        serde_json::from_str(json)
            .map_err(|e| PgmError::InvalidGraph(format!("JSON deserialization failed: {}", e)))
    }

    /// Compute mutual information between two variables using QuantRS format.
    pub fn mutual_information(joint: &DistributionExport, _var1: &str, _var2: &str) -> Result<f64> {
        if joint.variables.len() != 2 {
            return Err(PgmError::InvalidGraph(
                "Joint distribution must have exactly 2 variables".to_string(),
            ));
        }

        let mut mi = 0.0;
        let n1 = joint.cardinalities[0];
        let n2 = joint.cardinalities[1];

        // Compute marginals
        let mut p_x = vec![0.0; n1];
        let mut p_y = vec![0.0; n2];

        for (i, px) in p_x.iter_mut().enumerate().take(n1) {
            for (j, py) in p_y.iter_mut().enumerate().take(n2) {
                let idx = i * n2 + j;
                *px += joint.probabilities[idx];
                *py += joint.probabilities[idx];
            }
        }

        // Compute MI
        for (i, &px_val) in p_x.iter().enumerate().take(n1) {
            for (j, &py_val) in p_y.iter().enumerate().take(n2) {
                let idx = i * n2 + j;
                let p_xy = joint.probabilities[idx];
                if p_xy > 1e-10 && px_val > 1e-10 && py_val > 1e-10 {
                    mi += p_xy * (p_xy / (px_val * py_val)).ln();
                }
            }
        }

        Ok(mi)
    }

    /// Compute KL divergence between two distributions.
    pub fn kl_divergence(p: &DistributionExport, q: &DistributionExport) -> Result<f64> {
        if p.shape != q.shape {
            return Err(PgmError::InvalidGraph(
                "Distributions must have same shape".to_string(),
            ));
        }

        let mut kl = 0.0;
        for i in 0..p.probabilities.len() {
            let pi = p.probabilities[i];
            let qi = q.probabilities[i];

            if pi > 1e-10 {
                if qi < 1e-10 {
                    return Ok(f64::INFINITY);
                }
                kl += pi * (pi / qi).ln();
            }
        }

        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FactorGraph;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_factor_to_quantrs_distribution() {
        let values = Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        let dist = factor.to_quantrs_distribution().unwrap();

        assert_eq!(dist.variables.len(), 2);
        assert_eq!(dist.probabilities.len(), 4);
        assert!(dist.metadata.normalized);
    }

    #[test]
    fn test_quantrs_distribution_roundtrip() {
        let values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new("test".to_string(), vec!["x".to_string()], values).unwrap();

        let dist = factor.to_quantrs_distribution().unwrap();
        let factor2 = Factor::from_quantrs_distribution(&dist).unwrap();

        assert_eq!(factor.variables, factor2.variables);
        assert_eq!(factor.values.shape(), factor2.values.shape());
    }

    #[test]
    fn test_is_normalized() {
        let values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
            .unwrap()
            .into_dyn();
        let factor = Factor::new("test".to_string(), vec!["x".to_string()], values).unwrap();

        assert!(factor.is_normalized());
    }

    #[test]
    fn test_support() {
        let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        let support = factor.support();
        assert_eq!(support.len(), 4);
        assert_eq!(support[0], vec![0, 0]);
        assert_eq!(support[1], vec![0, 1]);
        assert_eq!(support[2], vec![1, 0]);
        assert_eq!(support[3], vec![1, 1]);
    }

    #[test]
    fn test_factor_graph_to_quantrs_model() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let factor = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(factor).unwrap();

        let model = graph.to_quantrs_model().unwrap();

        assert_eq!(model.variables.len(), 2);
        assert_eq!(model.factors.len(), 1);
        assert_eq!(model.model_type, "factor_graph");
    }

    #[test]
    fn test_model_stats() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let factor = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(factor).unwrap();

        let stats = graph.model_stats();

        assert_eq!(stats.num_variables, 2);
        assert_eq!(stats.num_factors, 1);
        assert_abs_diff_eq!(stats.avg_factor_size, 2.0);
        assert_eq!(stats.max_factor_size, 2);
    }

    #[test]
    fn test_mutual_information() {
        let dist = DistributionExport {
            variables: vec!["x".to_string(), "y".to_string()],
            cardinalities: vec![2, 2],
            probabilities: vec![0.25, 0.25, 0.25, 0.25],
            shape: vec![2, 2],
            metadata: DistributionMetadata {
                distribution_type: "categorical".to_string(),
                normalized: true,
                parameter_names: vec![],
                tags: vec![],
            },
        };

        let mi = utils::mutual_information(&dist, "x", "y").unwrap();

        assert_abs_diff_eq!(mi, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kl_divergence() {
        let p = DistributionExport {
            variables: vec!["x".to_string()],
            cardinalities: vec![2],
            probabilities: vec![0.7, 0.3],
            shape: vec![2],
            metadata: DistributionMetadata {
                distribution_type: "categorical".to_string(),
                normalized: true,
                parameter_names: vec![],
                tags: vec![],
            },
        };

        let q = DistributionExport {
            variables: vec!["x".to_string()],
            cardinalities: vec![2],
            probabilities: vec![0.5, 0.5],
            shape: vec![2],
            metadata: DistributionMetadata {
                distribution_type: "categorical".to_string(),
                normalized: true,
                parameter_names: vec![],
                tags: vec![],
            },
        };

        let kl = utils::kl_divergence(&p, &q).unwrap();

        assert!(kl > 0.0);
    }
}
