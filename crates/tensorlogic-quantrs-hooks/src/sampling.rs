//! Sampling-based inference methods for PGM.
//!
//! This module provides MCMC, importance sampling, and particle filtering algorithms
//! for approximate inference in probabilistic graphical models.
//!
//! # Algorithms
//!
//! - **Gibbs Sampling**: MCMC method for sampling from joint distributions
//! - **Importance Sampling**: Weighted sampling with proposal distributions
//! - **Particle Filter**: Sequential Monte Carlo for temporal models

use scirs2_core::ndarray::ArrayD;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;

/// Assignment of values to variables.
pub type Assignment = HashMap<String, usize>;

/// Gibbs sampling for approximate inference.
///
/// Uses Markov Chain Monte Carlo to sample from the joint distribution.
pub struct GibbsSampler {
    /// Number of burn-in samples to discard
    pub burn_in: usize,
    /// Number of samples to collect
    pub num_samples: usize,
    /// Thinning interval (keep every N-th sample)
    pub thinning: usize,
}

impl Default for GibbsSampler {
    fn default() -> Self {
        Self {
            burn_in: 100,
            num_samples: 1000,
            thinning: 1,
        }
    }
}

impl GibbsSampler {
    /// Create with custom parameters.
    pub fn new(burn_in: usize, num_samples: usize, thinning: usize) -> Self {
        Self {
            burn_in,
            num_samples,
            thinning,
        }
    }

    /// Run Gibbs sampling to approximate marginals.
    pub fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize random assignment
        let mut current_assignment = self.initialize_assignment(graph)?;

        // Burn-in phase
        for _ in 0..self.burn_in {
            self.gibbs_step(graph, &mut current_assignment)?;
        }

        // Collect samples
        let mut samples = Vec::new();
        for i in 0..self.num_samples * self.thinning {
            self.gibbs_step(graph, &mut current_assignment)?;

            // Keep sample if it's at thinning interval
            if i % self.thinning == 0 {
                samples.push(current_assignment.clone());
            }
        }

        // Compute empirical marginals from samples
        self.compute_empirical_marginals(graph, &samples)
    }

    /// Initialize random assignment for all variables.
    fn initialize_assignment(&self, graph: &FactorGraph) -> Result<Assignment> {
        let mut rng = thread_rng();
        let mut assignment = Assignment::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let random_value = rng.gen_range(0..var_node.cardinality);
                assignment.insert(var_name.clone(), random_value);
            }
        }

        Ok(assignment)
    }

    /// Perform one Gibbs sampling step (resample all variables).
    fn gibbs_step(&self, graph: &FactorGraph, assignment: &mut Assignment) -> Result<()> {
        // Resample each variable conditioned on others
        for var_name in graph.variable_names() {
            self.resample_variable(graph, var_name, assignment)?;
        }

        Ok(())
    }

    /// Resample a single variable given current assignment of others.
    fn resample_variable(
        &self,
        graph: &FactorGraph,
        var_name: &str,
        assignment: &mut Assignment,
    ) -> Result<()> {
        let var_node = graph
            .get_variable(var_name)
            .ok_or_else(|| PgmError::VariableNotFound(var_name.to_string()))?;

        // Compute conditional distribution P(X | others)
        let mut conditional_probs = vec![0.0; var_node.cardinality];

        for (value, prob) in conditional_probs
            .iter_mut()
            .enumerate()
            .take(var_node.cardinality)
        {
            assignment.insert(var_name.to_string(), value);
            *prob = self.compute_joint_probability(graph, assignment)?;
        }

        // Normalize
        let sum: f64 = conditional_probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut conditional_probs {
                *prob /= sum;
            }
        } else {
            // Fallback to uniform if all zero
            let uniform_prob = 1.0 / var_node.cardinality as f64;
            conditional_probs = vec![uniform_prob; var_node.cardinality];
        }

        // Sample from conditional distribution
        let sampled_value = self.sample_from_distribution(&conditional_probs);
        assignment.insert(var_name.to_string(), sampled_value);

        Ok(())
    }

    /// Compute joint probability for a full assignment.
    fn compute_joint_probability(
        &self,
        graph: &FactorGraph,
        assignment: &Assignment,
    ) -> Result<f64> {
        let mut prob = 1.0;

        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                // Build index for this factor
                let mut indices = Vec::new();
                for var in &factor.variables {
                    if let Some(&value) = assignment.get(var) {
                        indices.push(value);
                    } else {
                        return Err(PgmError::VariableNotFound(var.clone()));
                    }
                }

                prob *= factor.values[indices.as_slice()];
            }
        }

        Ok(prob)
    }

    /// Sample from a discrete probability distribution.
    fn sample_from_distribution(&self, probs: &[f64]) -> usize {
        let mut rng = thread_rng();
        let u: f64 = rng.random();

        let mut cumulative = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if u < cumulative {
                return idx;
            }
        }

        // Fallback to last index
        probs.len() - 1
    }

    /// Compute empirical marginals from collected samples.
    fn compute_empirical_marginals(
        &self,
        graph: &FactorGraph,
        samples: &[Assignment],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut marginals = HashMap::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut counts = vec![0; var_node.cardinality];

                // Count occurrences
                for sample in samples {
                    if let Some(&value) = sample.get(var_name) {
                        counts[value] += 1;
                    }
                }

                // Normalize to probabilities
                let total = samples.len() as f64;
                let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();

                marginals.insert(
                    var_name.clone(),
                    ArrayD::from_shape_vec(vec![var_node.cardinality], probs)?,
                );
            }
        }

        Ok(marginals)
    }

    /// Get all samples (for analysis).
    pub fn get_samples(&self, graph: &FactorGraph) -> Result<Vec<Assignment>> {
        let mut current_assignment = self.initialize_assignment(graph)?;

        // Burn-in
        for _ in 0..self.burn_in {
            self.gibbs_step(graph, &mut current_assignment)?;
        }

        // Collect samples
        let mut samples = Vec::new();
        for i in 0..self.num_samples * self.thinning {
            self.gibbs_step(graph, &mut current_assignment)?;

            if i % self.thinning == 0 {
                samples.push(current_assignment.clone());
            }
        }

        Ok(samples)
    }
}

impl From<scirs2_core::ndarray::ShapeError> for PgmError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        PgmError::InvalidDistribution(format!("Shape error: {}", err))
    }
}

/// Weighted sample for importance sampling.
#[derive(Debug, Clone)]
pub struct WeightedSample {
    /// The assignment of values to variables
    pub assignment: Assignment,
    /// The unnormalized importance weight
    pub weight: f64,
    /// Log weight for numerical stability
    pub log_weight: f64,
}

/// Importance sampling for approximate inference.
///
/// Importance sampling draws samples from a proposal distribution q(x)
/// and weights them by p(x)/q(x) to estimate expectations under p(x).
///
/// # Example
///
/// ```
/// use tensorlogic_quantrs_hooks::{FactorGraph, ImportanceSampler, ProposalDistribution};
///
/// let mut graph = FactorGraph::new();
/// graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
///
/// let sampler = ImportanceSampler::new(1000);
/// let result = sampler.run(&graph, ProposalDistribution::Uniform);
/// ```
pub struct ImportanceSampler {
    /// Number of samples to draw
    pub num_samples: usize,
    /// Whether to use self-normalized importance sampling
    pub self_normalize: bool,
}

/// Proposal distribution types for importance sampling.
#[derive(Debug, Clone)]
pub enum ProposalDistribution {
    /// Uniform distribution over all states
    Uniform,
    /// Custom proposal weights (not normalized)
    Custom(HashMap<String, Vec<f64>>),
    /// Prior distribution from the model
    Prior,
}

impl Default for ImportanceSampler {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            self_normalize: true,
        }
    }
}

impl ImportanceSampler {
    /// Create a new importance sampler with specified number of samples.
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            self_normalize: true,
        }
    }

    /// Set whether to use self-normalized importance sampling.
    pub fn with_self_normalize(mut self, self_normalize: bool) -> Self {
        self.self_normalize = self_normalize;
        self
    }

    /// Run importance sampling to approximate marginals.
    pub fn run(
        &self,
        graph: &FactorGraph,
        proposal: ProposalDistribution,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let samples = self.draw_weighted_samples(graph, &proposal)?;
        self.compute_weighted_marginals(graph, &samples)
    }

    /// Draw weighted samples from the proposal distribution.
    pub fn draw_weighted_samples(
        &self,
        graph: &FactorGraph,
        proposal: &ProposalDistribution,
    ) -> Result<Vec<WeightedSample>> {
        let mut samples = Vec::with_capacity(self.num_samples);
        let mut rng = thread_rng();

        for _ in 0..self.num_samples {
            // Sample from proposal
            let (assignment, proposal_prob) =
                self.sample_from_proposal(graph, proposal, &mut rng)?;

            // Compute target probability
            let target_prob = self.compute_target_probability(graph, &assignment)?;

            // Compute importance weight
            let weight = if proposal_prob > 0.0 {
                target_prob / proposal_prob
            } else {
                0.0
            };

            let log_weight = if proposal_prob > 0.0 && target_prob > 0.0 {
                target_prob.ln() - proposal_prob.ln()
            } else {
                f64::NEG_INFINITY
            };

            samples.push(WeightedSample {
                assignment,
                weight,
                log_weight,
            });
        }

        Ok(samples)
    }

    /// Sample from the proposal distribution.
    fn sample_from_proposal(
        &self,
        graph: &FactorGraph,
        proposal: &ProposalDistribution,
        rng: &mut impl Rng,
    ) -> Result<(Assignment, f64)> {
        let mut assignment = Assignment::new();
        let mut proposal_prob = 1.0;

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let (value, prob) = match proposal {
                    ProposalDistribution::Uniform => {
                        let value = rng.random_range(0..var_node.cardinality);
                        let prob = 1.0 / var_node.cardinality as f64;
                        (value, prob)
                    }
                    ProposalDistribution::Custom(weights) => {
                        if let Some(var_weights) = weights.get(var_name) {
                            let (value, prob) = self.sample_categorical(var_weights, rng);
                            (value, prob)
                        } else {
                            let value = rng.random_range(0..var_node.cardinality);
                            let prob = 1.0 / var_node.cardinality as f64;
                            (value, prob)
                        }
                    }
                    ProposalDistribution::Prior => {
                        // Use uniform for now; could be extended to use prior factors
                        let value = rng.random_range(0..var_node.cardinality);
                        let prob = 1.0 / var_node.cardinality as f64;
                        (value, prob)
                    }
                };

                assignment.insert(var_name.clone(), value);
                proposal_prob *= prob;
            }
        }

        Ok((assignment, proposal_prob))
    }

    /// Sample from a categorical distribution given weights.
    fn sample_categorical(&self, weights: &[f64], rng: &mut impl Rng) -> (usize, f64) {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return (0, 1.0 / weights.len() as f64);
        }

        let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let u: f64 = rng.random();

        let mut cumulative = 0.0;
        for (idx, &prob) in normalized.iter().enumerate() {
            cumulative += prob;
            if u < cumulative {
                return (idx, prob);
            }
        }

        (weights.len() - 1, *normalized.last().unwrap_or(&0.0))
    }

    /// Compute target probability (unnormalized) for an assignment.
    fn compute_target_probability(
        &self,
        graph: &FactorGraph,
        assignment: &Assignment,
    ) -> Result<f64> {
        let mut prob = 1.0;

        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                let mut indices = Vec::new();
                for var in &factor.variables {
                    if let Some(&value) = assignment.get(var) {
                        indices.push(value);
                    } else {
                        return Err(PgmError::VariableNotFound(var.clone()));
                    }
                }
                prob *= factor.values[indices.as_slice()];
            }
        }

        Ok(prob)
    }

    /// Compute weighted marginals from importance samples.
    fn compute_weighted_marginals(
        &self,
        graph: &FactorGraph,
        samples: &[WeightedSample],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut marginals = HashMap::new();

        // Compute total weight for self-normalization
        let total_weight: f64 = samples.iter().map(|s| s.weight).sum();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut weighted_counts = vec![0.0; var_node.cardinality];

                // Accumulate weighted counts
                for sample in samples {
                    if let Some(&value) = sample.assignment.get(var_name) {
                        weighted_counts[value] += sample.weight;
                    }
                }

                // Normalize
                let probs: Vec<f64> = if self.self_normalize && total_weight > 0.0 {
                    weighted_counts.iter().map(|&c| c / total_weight).collect()
                } else {
                    let sum: f64 = weighted_counts.iter().sum();
                    if sum > 0.0 {
                        weighted_counts.iter().map(|&c| c / sum).collect()
                    } else {
                        vec![1.0 / var_node.cardinality as f64; var_node.cardinality]
                    }
                };

                marginals.insert(
                    var_name.clone(),
                    ArrayD::from_shape_vec(vec![var_node.cardinality], probs)?,
                );
            }
        }

        Ok(marginals)
    }

    /// Get all weighted samples for analysis.
    pub fn get_weighted_samples(
        &self,
        graph: &FactorGraph,
        proposal: &ProposalDistribution,
    ) -> Result<Vec<WeightedSample>> {
        self.draw_weighted_samples(graph, proposal)
    }

    /// Compute the effective sample size (ESS).
    ///
    /// ESS measures the efficiency of importance sampling.
    /// Higher ESS indicates better proposal distribution.
    pub fn effective_sample_size(samples: &[WeightedSample]) -> f64 {
        let weights: Vec<f64> = samples.iter().map(|s| s.weight).collect();
        let sum_w: f64 = weights.iter().sum();
        let sum_w2: f64 = weights.iter().map(|w| w * w).sum();

        if sum_w2 > 0.0 {
            (sum_w * sum_w) / sum_w2
        } else {
            0.0
        }
    }

    /// Compute the coefficient of variation of weights.
    pub fn weight_coefficient_of_variation(samples: &[WeightedSample]) -> f64 {
        let n = samples.len() as f64;
        let weights: Vec<f64> = samples.iter().map(|s| s.weight).collect();
        let mean = weights.iter().sum::<f64>() / n;
        let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if mean > 0.0 {
            std_dev / mean
        } else {
            0.0
        }
    }

    /// Resample particles based on their weights (for particle filtering).
    pub fn resample(samples: &[WeightedSample]) -> Vec<WeightedSample> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }

        let mut rng = thread_rng();
        let total_weight: f64 = samples.iter().map(|s| s.weight).sum();

        if total_weight <= 0.0 {
            return samples.to_vec();
        }

        let normalized_weights: Vec<f64> =
            samples.iter().map(|s| s.weight / total_weight).collect();

        // Systematic resampling
        let mut resampled = Vec::with_capacity(n);
        let u0: f64 = rng.random::<f64>() / n as f64;

        let mut cumulative = 0.0;
        let mut j = 0;

        for i in 0..n {
            let u = u0 + (i as f64) / (n as f64);
            while cumulative + normalized_weights[j] < u && j < n - 1 {
                cumulative += normalized_weights[j];
                j += 1;
            }

            resampled.push(WeightedSample {
                assignment: samples[j].assignment.clone(),
                weight: 1.0,
                log_weight: 0.0,
            });
        }

        resampled
    }
}

/// Particle for particle filtering.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current state assignment
    pub state: Assignment,
    /// Particle weight
    pub weight: f64,
    /// Log weight for numerical stability
    pub log_weight: f64,
    /// History of states (optional)
    pub history: Vec<Assignment>,
}

/// Particle filter (Sequential Monte Carlo) for temporal inference.
///
/// Particle filtering is used for inference in dynamic systems
/// where the state evolves over time.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_quantrs_hooks::{ParticleFilter, HiddenMarkovModel, Assignment};
/// use std::collections::HashMap;
///
/// // Create HMM with 2 states, 3 observations, 10 time steps
/// let hmm = HiddenMarkovModel::new(2, 3, 10);
///
/// // Create particle filter
/// let mut pf = ParticleFilter::new(100, vec!["state".to_string()]);
///
/// // Initialize particles
/// let cardinalities: HashMap<String, usize> = [("state".to_string(), 2)].into_iter().collect();
/// pf.initialize(&cardinalities);
/// ```
pub struct ParticleFilter {
    /// Number of particles
    pub num_particles: usize,
    /// Current particles
    pub particles: Vec<Particle>,
    /// State variable names
    pub state_variables: Vec<String>,
    /// Effective sample size threshold for resampling
    pub ess_threshold: f64,
    /// Whether to track history
    pub track_history: bool,
}

impl ParticleFilter {
    /// Create a new particle filter.
    pub fn new(num_particles: usize, state_variables: Vec<String>) -> Self {
        Self {
            num_particles,
            particles: Vec::new(),
            state_variables,
            ess_threshold: 0.5,
            track_history: false,
        }
    }

    /// Set the ESS threshold for resampling (as fraction of num_particles).
    pub fn with_ess_threshold(mut self, threshold: f64) -> Self {
        self.ess_threshold = threshold;
        self
    }

    /// Enable history tracking.
    pub fn with_history(mut self, track: bool) -> Self {
        self.track_history = track;
        self
    }

    /// Initialize particles uniformly.
    pub fn initialize(&mut self, cardinalities: &HashMap<String, usize>) {
        let mut rng = thread_rng();
        self.particles = Vec::with_capacity(self.num_particles);

        for _ in 0..self.num_particles {
            let mut state = Assignment::new();

            for var_name in &self.state_variables {
                if let Some(&card) = cardinalities.get(var_name) {
                    let value = rng.gen_range(0..card);
                    state.insert(var_name.clone(), value);
                }
            }

            self.particles.push(Particle {
                state,
                weight: 1.0 / self.num_particles as f64,
                log_weight: -(self.num_particles as f64).ln(),
                history: Vec::new(),
            });
        }
    }

    /// Initialize particles from a prior distribution.
    pub fn initialize_from_prior(&mut self, prior: &[f64], cardinalities: &HashMap<String, usize>) {
        let mut rng = thread_rng();
        self.particles = Vec::with_capacity(self.num_particles);

        let total: f64 = prior.iter().sum();
        let normalized: Vec<f64> = prior.iter().map(|p| p / total).collect();

        for _ in 0..self.num_particles {
            let mut state = Assignment::new();

            // Sample state from prior (assuming single state variable)
            if let Some(var_name) = self.state_variables.first() {
                let u: f64 = rng.random();
                let mut cumulative = 0.0;
                let mut value = 0;

                for (idx, &prob) in normalized.iter().enumerate() {
                    cumulative += prob;
                    if u < cumulative {
                        value = idx;
                        break;
                    }
                }

                state.insert(var_name.clone(), value);
            }

            // Initialize other variables uniformly
            for var_name in self.state_variables.iter().skip(1) {
                if let Some(&card) = cardinalities.get(var_name) {
                    let value = rng.gen_range(0..card);
                    state.insert(var_name.clone(), value);
                }
            }

            self.particles.push(Particle {
                state,
                weight: 1.0 / self.num_particles as f64,
                log_weight: -(self.num_particles as f64).ln(),
                history: Vec::new(),
            });
        }
    }

    /// Predict step: propagate particles through transition model.
    ///
    /// The transition function takes a state and a random seed, returning the next state.
    pub fn predict(
        &mut self,
        transition: &dyn Fn(&Assignment, u64) -> Assignment,
        cardinalities: &HashMap<String, usize>,
    ) {
        let mut rng = thread_rng();

        for particle in &mut self.particles {
            if self.track_history {
                particle.history.push(particle.state.clone());
            }

            // Generate a random seed for the transition
            let seed: u64 = rng.random();
            particle.state = transition(&particle.state, seed);

            // Ensure state values are within bounds
            for var_name in &self.state_variables {
                if let Some(&card) = cardinalities.get(var_name) {
                    if let Some(value) = particle.state.get_mut(var_name) {
                        *value = (*value).min(card.saturating_sub(1));
                    }
                }
            }
        }
    }

    /// Update step: weight particles based on observation likelihood.
    pub fn update<F>(&mut self, observation: &Assignment, likelihood: F)
    where
        F: Fn(&Assignment, &Assignment) -> f64,
    {
        // Compute likelihood for each particle
        for particle in &mut self.particles {
            let lik = likelihood(&particle.state, observation);
            particle.weight *= lik;
            if lik > 0.0 {
                particle.log_weight += lik.ln();
            } else {
                particle.log_weight = f64::NEG_INFINITY;
            }
        }

        // Normalize weights
        self.normalize_weights();

        // Resample if ESS is too low
        let ess = self.effective_sample_size();
        if ess < self.ess_threshold * self.num_particles as f64 {
            self.resample();
        }
    }

    /// Normalize particle weights.
    fn normalize_weights(&mut self) {
        let total: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= total;
            }
        }
    }

    /// Compute effective sample size.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_w2: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_w2 > 0.0 {
            1.0 / sum_w2
        } else {
            0.0
        }
    }

    /// Resample particles using systematic resampling.
    pub fn resample(&mut self) {
        let n = self.num_particles;
        let mut rng = thread_rng();

        // Build CDF
        let mut cdf = Vec::with_capacity(n);
        let mut cumulative = 0.0;
        for particle in &self.particles {
            cumulative += particle.weight;
            cdf.push(cumulative);
        }

        // Systematic resampling
        let u0: f64 = rng.random::<f64>() / n as f64;
        let mut new_particles = Vec::with_capacity(n);

        let mut j = 0;
        for i in 0..n {
            let u = u0 + (i as f64) / (n as f64);
            while j < n - 1 && cdf[j] < u {
                j += 1;
            }

            new_particles.push(Particle {
                state: self.particles[j].state.clone(),
                weight: 1.0 / n as f64,
                log_weight: -(n as f64).ln(),
                history: if self.track_history {
                    self.particles[j].history.clone()
                } else {
                    Vec::new()
                },
            });
        }

        self.particles = new_particles;
    }

    /// Estimate marginal distribution from particles.
    pub fn estimate_marginal(&self, var_name: &str, cardinality: usize) -> Vec<f64> {
        let mut counts = vec![0.0; cardinality];

        for particle in &self.particles {
            if let Some(&value) = particle.state.get(var_name) {
                if value < cardinality {
                    counts[value] += particle.weight;
                }
            }
        }

        // Normalize
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            counts.iter().map(|c| c / total).collect()
        } else {
            vec![1.0 / cardinality as f64; cardinality]
        }
    }

    /// Estimate expected value of a function over the particle distribution.
    pub fn estimate_expectation<F>(&self, func: F) -> f64
    where
        F: Fn(&Assignment) -> f64,
    {
        self.particles
            .iter()
            .map(|p| p.weight * func(&p.state))
            .sum()
    }

    /// Get the MAP (most likely) state.
    pub fn map_estimate(&self) -> Option<&Assignment> {
        self.particles
            .iter()
            .max_by(|a, b| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| &p.state)
    }

    /// Get entropy of the particle distribution.
    pub fn entropy(&self) -> f64 {
        self.particles
            .iter()
            .filter(|p| p.weight > 0.0)
            .map(|p| -p.weight * p.weight.ln())
            .sum()
    }

    /// Run particle filter on a sequence of observations.
    ///
    /// The transition function takes a state and a random seed.
    /// The likelihood function computes P(observation | state).
    pub fn run_sequence(
        &mut self,
        observations: &[Assignment],
        transition: &dyn Fn(&Assignment, u64) -> Assignment,
        likelihood: &dyn Fn(&Assignment, &Assignment) -> f64,
        cardinalities: &HashMap<String, usize>,
    ) -> Vec<Vec<f64>> {
        let mut marginals = Vec::with_capacity(observations.len());

        for obs in observations {
            // Predict
            self.predict(transition, cardinalities);

            // Update
            self.update(obs, likelihood);

            // Record marginal for first state variable
            if let Some(var_name) = self.state_variables.first() {
                if let Some(&card) = cardinalities.get(var_name) {
                    marginals.push(self.estimate_marginal(var_name, card));
                }
            }
        }

        marginals
    }
}

/// Likelihood weighting for Bayesian networks.
///
/// A specialized form of importance sampling where:
/// - Sample non-evidence variables from prior
/// - Weight by likelihood of evidence
pub struct LikelihoodWeighting {
    /// Number of samples
    pub num_samples: usize,
}

impl Default for LikelihoodWeighting {
    fn default() -> Self {
        Self { num_samples: 1000 }
    }
}

impl LikelihoodWeighting {
    /// Create a new likelihood weighting sampler.
    pub fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }

    /// Run likelihood weighting with evidence.
    pub fn run(
        &self,
        graph: &FactorGraph,
        evidence: &Assignment,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut weighted_samples = Vec::with_capacity(self.num_samples);
        let mut rng = thread_rng();

        for _ in 0..self.num_samples {
            let (assignment, weight) = self.sample_with_evidence(graph, evidence, &mut rng)?;

            weighted_samples.push(WeightedSample {
                assignment,
                weight,
                log_weight: if weight > 0.0 {
                    weight.ln()
                } else {
                    f64::NEG_INFINITY
                },
            });
        }

        // Compute marginals
        let sampler = ImportanceSampler::new(self.num_samples);
        sampler.compute_weighted_marginals(graph, &weighted_samples)
    }

    /// Sample non-evidence variables and compute weight from evidence.
    fn sample_with_evidence(
        &self,
        graph: &FactorGraph,
        evidence: &Assignment,
        rng: &mut impl Rng,
    ) -> Result<(Assignment, f64)> {
        let mut assignment = Assignment::new();
        let mut weight = 1.0;

        // Set evidence variables
        for (var, value) in evidence {
            assignment.insert(var.clone(), *value);
        }

        // Sample non-evidence variables uniformly
        for var_name in graph.variable_names() {
            if !evidence.contains_key(var_name) {
                if let Some(var_node) = graph.get_variable(var_name) {
                    let value = rng.random_range(0..var_node.cardinality);
                    assignment.insert(var_name.clone(), value);
                }
            }
        }

        // Compute weight as product of factors
        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                let mut indices = Vec::new();
                for var in &factor.variables {
                    if let Some(&value) = assignment.get(var) {
                        indices.push(value);
                    } else {
                        return Err(PgmError::VariableNotFound(var.clone()));
                    }
                }
                weight *= factor.values[indices.as_slice()];
            }
        }

        Ok((assignment, weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gibbs_sampler_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 100, 1);
        let result = sampler.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));

        // Should be approximately uniform
        let dist = &marginals["x"];
        let sum: f64 = dist.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gibbs_sampler_multiple_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(20, 100, 1);
        let result = sampler.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert_eq!(marginals.len(), 2);
    }

    #[test]
    fn test_sample_collection() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 50, 1);
        let samples = sampler.get_samples(&graph);
        assert!(samples.is_ok());
        assert_eq!(samples.unwrap().len(), 50);
    }

    #[test]
    fn test_gibbs_with_thinning() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 50, 5);
        let samples = sampler.get_samples(&graph);
        assert!(samples.is_ok());
        assert_eq!(samples.unwrap().len(), 50);
    }

    #[test]
    fn test_importance_sampler_uniform() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = ImportanceSampler::new(100);
        let result = sampler.run(&graph, ProposalDistribution::Uniform);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));

        let dist = &marginals["x"];
        let sum: f64 = dist.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_importance_sampler_custom_proposal() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mut custom_weights = HashMap::new();
        custom_weights.insert("x".to_string(), vec![0.8, 0.2]);

        let sampler = ImportanceSampler::new(100);
        let result = sampler.run(&graph, ProposalDistribution::Custom(custom_weights));
        assert!(result.is_ok());

        let marginals = result.unwrap();
        let sum: f64 = marginals["x"].iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_effective_sample_size() {
        let samples = vec![
            WeightedSample {
                assignment: HashMap::new(),
                weight: 0.5,
                log_weight: 0.5_f64.ln(),
            },
            WeightedSample {
                assignment: HashMap::new(),
                weight: 0.5,
                log_weight: 0.5_f64.ln(),
            },
        ];

        let ess = ImportanceSampler::effective_sample_size(&samples);
        // Equal weights should give ESS = N
        assert_abs_diff_eq!(ess, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_particle_filter_initialization() {
        let mut pf = ParticleFilter::new(10, vec!["state".to_string()]);
        let cardinalities: HashMap<String, usize> =
            [("state".to_string(), 3)].into_iter().collect();
        pf.initialize(&cardinalities);

        assert_eq!(pf.particles.len(), 10);

        // All particles should have equal weight
        for particle in &pf.particles {
            assert_abs_diff_eq!(particle.weight, 0.1, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_particle_filter_estimate_marginal() {
        let mut pf = ParticleFilter::new(100, vec!["state".to_string()]);
        let cardinalities: HashMap<String, usize> =
            [("state".to_string(), 2)].into_iter().collect();
        pf.initialize(&cardinalities);

        let marginal = pf.estimate_marginal("state", 2);
        assert_eq!(marginal.len(), 2);

        // Should sum to 1
        let sum: f64 = marginal.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_particle_filter_ess() {
        let mut pf = ParticleFilter::new(100, vec!["state".to_string()]);
        let cardinalities: HashMap<String, usize> =
            [("state".to_string(), 2)].into_iter().collect();
        pf.initialize(&cardinalities);

        let ess = pf.effective_sample_size();
        // Uniform weights should give ESS close to N
        assert!(ess > 90.0);
    }

    #[test]
    fn test_particle_filter_resample() {
        let mut pf = ParticleFilter::new(10, vec!["state".to_string()]);
        let cardinalities: HashMap<String, usize> =
            [("state".to_string(), 2)].into_iter().collect();
        pf.initialize(&cardinalities);

        // Manually set unequal weights
        for (i, particle) in pf.particles.iter_mut().enumerate() {
            particle.weight = if i == 0 { 1.0 } else { 0.0 };
        }

        pf.resample();

        // After resampling, weights should be equal
        for particle in &pf.particles {
            assert_abs_diff_eq!(particle.weight, 0.1, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_likelihood_weighting() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let mut evidence = Assignment::new();
        evidence.insert("y".to_string(), 1);

        let lw = LikelihoodWeighting::new(100);
        let result = lw.run(&graph, &evidence);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));
    }

    #[test]
    fn test_importance_sampler_weighted_samples() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = ImportanceSampler::new(50);
        let samples = sampler
            .get_weighted_samples(&graph, &ProposalDistribution::Uniform)
            .unwrap();

        assert_eq!(samples.len(), 50);

        // All samples should have valid assignments
        for sample in &samples {
            assert!(sample.assignment.contains_key("x"));
        }
    }

    #[test]
    fn test_weight_coefficient_of_variation() {
        let samples = vec![
            WeightedSample {
                assignment: HashMap::new(),
                weight: 1.0,
                log_weight: 0.0,
            },
            WeightedSample {
                assignment: HashMap::new(),
                weight: 1.0,
                log_weight: 0.0,
            },
        ];

        let cv = ImportanceSampler::weight_coefficient_of_variation(&samples);
        // Equal weights should give CV = 0
        assert_abs_diff_eq!(cv, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_particle_filter_with_history() {
        let pf = ParticleFilter::new(5, vec!["state".to_string()])
            .with_history(true)
            .with_ess_threshold(0.3);

        assert!(pf.track_history);
        assert_abs_diff_eq!(pf.ess_threshold, 0.3, epsilon = 1e-6);
    }
}
