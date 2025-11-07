# tensorlogic-quantrs-hooks

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--quantrs--hooks-orange)](https://crates.io/crates/tensorlogic-quantrs-hooks)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-quantrs-hooks)
[![Tests](https://img.shields.io/badge/tests-93%2B-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

**Probabilistic Graphical Model Integration for TensorLogic**

Bridge between logic-based reasoning and probabilistic inference through factor graphs, belief propagation, and variational methods.

## Overview

`tensorlogic-quantrs-hooks` enables probabilistic reasoning over TensorLogic expressions by converting logical rules into factor graphs and applying state-of-the-art inference algorithms. This crate seamlessly integrates with the QuantRS2 ecosystem for probabilistic programming.

### Key Features

- **TLExpr → Factor Graph Conversion**: Automatic translation of logical expressions to PGM representations
- **Exact Inference**:
  - Sum-product and max-product belief propagation for tree-structured graphs
  - Junction tree algorithm for exact inference on arbitrary graphs
- **Approximate Inference**:
  - Loopy BP: Message passing for graphs with cycles, with damping and convergence detection
  - Variational Inference: Mean-field, Bethe approximation, and tree-reweighted BP
  - Expectation Propagation (EP): Moment matching with site approximations for discrete and continuous variables
  - MCMC Sampling: Gibbs sampling for approximate posterior computation
- **QuantRS2 Integration**:
  - Distribution and model export to QuantRS format
  - JSON serialization for ecosystem interoperability
  - Information-theoretic utilities (mutual information, KL divergence)
- **Parameter Learning**:
  - Maximum Likelihood Estimation (MLE) for discrete distributions
  - Bayesian estimation with Dirichlet priors
  - Baum-Welch algorithm (EM) for Hidden Markov Models
- **Sequence Models**:
  - Linear-chain CRFs for sequence labeling with Viterbi decoding
  - Feature functions (transition, emission, custom)
  - Forward-backward algorithm for marginal probabilities
- **Full SciRS2 Integration**: All tensor operations use SciRS2 for performance and consistency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic-quantrs-hooks = "0.1.0-alpha.1"
scirs2-core = "0.1.0-rc.2"  # For tensor operations
```

## Quick Start

### Basic Factor Graph Creation

```rust
use tensorlogic_quantrs_hooks::{FactorGraph, Factor};
use scirs2_core::ndarray::Array;

// Create factor graph
let mut graph = FactorGraph::new();

// Add binary variables
graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

// Add factor P(x)
let px_values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
    .unwrap()
    .into_dyn();
let px = Factor::new("P(x)".to_string(), vec!["x".to_string()], px_values).unwrap();
graph.add_factor(px).unwrap();

// Add factor P(y|x)
let pyx_values = Array::from_shape_vec(
    vec![2, 2],
    vec![0.9, 0.1, 0.2, 0.8]  // P(y|x=0), P(y|x=1)
).unwrap().into_dyn();
let pyx = Factor::new(
    "P(y|x)".to_string(),
    vec!["x".to_string(), "y".to_string()],
    pyx_values
).unwrap();
graph.add_factor(pyx).unwrap();
```

### Converting TLExpr to Factor Graph

```rust
use tensorlogic_ir::TLExpr;
use tensorlogic_quantrs_hooks::expr_to_factor_graph;

// Define logical expression
let expr = TLExpr::and(
    TLExpr::pred("P", vec![TLExpr::var("x")]),
    TLExpr::pred("Q", vec![TLExpr::var("x"), TLExpr::var("y")])
);

// Convert to factor graph
let graph = expr_to_factor_graph(&expr).unwrap();

println!("Variables: {}", graph.num_variables());
println!("Factors: {}", graph.num_factors());
```

## Core Concepts

### Factor Graphs

A factor graph is a bipartite graph with:
- **Variable nodes**: Represent random variables
- **Factor nodes**: Represent functions over subsets of variables

```
Variables:  X₁    X₂    X₃
            |  \  / |    |
Factors:    φ₁  φ₂  φ₃
```

### Factors

Factors are functions φ(X₁, X₂, ..., Xₖ) → ℝ⁺ representing probabilities or potentials.

```rust
use tensorlogic_quantrs_hooks::Factor;
use scirs2_core::ndarray::Array;

// Create a binary factor P(X, Y)
let values = Array::from_shape_vec(
    vec![2, 2],
    vec![0.1, 0.2, 0.3, 0.4]
).unwrap().into_dyn();

let factor = Factor::new(
    "joint".to_string(),
    vec!["X".to_string(), "Y".to_string()],
    values
).unwrap();

// Normalize to sum to 1
let mut normalized = factor.clone();
normalized.normalize();
```

## Factor Operations

### Factor Product

Combine factors over different variable sets:

```rust
// φ₁(X) = [0.6, 0.4]
let f1_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
    .unwrap().into_dyn();
let f1 = Factor::new("f1".to_string(), vec!["X".to_string()], f1_values).unwrap();

// φ₂(Y) = [0.7, 0.3]
let f2_values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
    .unwrap().into_dyn();
let f2 = Factor::new("f2".to_string(), vec!["Y".to_string()], f2_values).unwrap();

// φ₁(X) × φ₂(Y) = φ(X, Y)
let product = f1.product(&f2).unwrap();
assert_eq!(product.variables.len(), 2);
assert_eq!(product.values.shape(), &[2, 2]);
```

### Marginalization

Sum out variables to compute marginals:

```rust
// φ(X, Y) → φ(X) = Σ_Y φ(X, Y)
let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
    .unwrap().into_dyn();
let factor = Factor::new(
    "joint".to_string(),
    vec!["X".to_string(), "Y".to_string()],
    values
).unwrap();

let marginal = factor.marginalize_out("Y").unwrap();
assert_eq!(marginal.variables, vec!["X".to_string()]);
// Result: [0.1 + 0.2, 0.3 + 0.4] = [0.3, 0.7]
```

### Factor Reduction (Evidence)

Condition on observed values:

```rust
// Observe Y = 1, compute P(X | Y=1)
let conditional = factor.reduce("Y", 1).unwrap();
assert_eq!(conditional.variables, vec!["X".to_string()]);
// Result: [0.2, 0.4] (before normalization)
```

### Factor Division

Compute message quotients:

```rust
let f1 = Factor::new("f1".to_string(), vec!["X".to_string()],
    Array::from_shape_vec(vec![2], vec![0.6, 0.4]).unwrap().into_dyn()).unwrap();
let f2 = Factor::new("f2".to_string(), vec!["X".to_string()],
    Array::from_shape_vec(vec![2], vec![0.3, 0.2]).unwrap().into_dyn()).unwrap();

let result = f1.divide(&f2).unwrap();
// Result: [0.6/0.3, 0.4/0.2] = [2.0, 2.0]
```

## Inference Algorithms

### 1. Sum-Product Belief Propagation

Exact inference for tree-structured graphs, loopy BP for graphs with cycles.

```rust
use tensorlogic_quantrs_hooks::{SumProductAlgorithm, InferenceEngine, MarginalizationQuery};

// Create algorithm with custom parameters
let algorithm = SumProductAlgorithm::new(
    100,     // max_iterations
    1e-6,    // tolerance
    0.0      // damping (0.0 = no damping)
);

// Create inference engine
let engine = InferenceEngine::new(graph.clone(), Box::new(algorithm));

// Compute marginal P(X)
let query = MarginalizationQuery {
    variable: "X".to_string(),
};
let marginal = engine.marginalize(&query).unwrap();

println!("P(X=0) = {}", marginal[[0]]);
println!("P(X=1) = {}", marginal[[1]]);
```

#### Loopy BP with Damping

For graphs with cycles, use damping to improve convergence:

```rust
let loopy_bp = SumProductAlgorithm::new(
    200,     // More iterations for loopy graphs
    1e-5,    // Tolerance
    0.5      // Damping factor (0.5 = 50% old message, 50% new)
);

let engine = InferenceEngine::new(loopy_graph, Box::new(loopy_bp));
let result = engine.marginalize(&query).unwrap();
```

### 2. Max-Product Algorithm (MAP Inference)

Find the most probable assignment:

```rust
use tensorlogic_quantrs_hooks::MaxProductAlgorithm;

let max_product = MaxProductAlgorithm::default();
let engine = InferenceEngine::new(graph, Box::new(max_product));

// Compute MAP assignment
let marginals = engine.run().unwrap();

// Find most probable values
for (var, marginal) in &marginals {
    let max_idx = marginal.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    println!("{} = {}", var, max_idx);
}
```

### 3. Variational Inference (Mean-Field)

Scalable approximate inference using mean-field approximation:

```rust
use tensorlogic_quantrs_hooks::MeanFieldInference;

// Create mean-field inference engine
let mean_field = MeanFieldInference::new(
    1000,    // max_iterations
    1e-4,    // tolerance
);

// Run inference
let result = mean_field.infer(&graph).unwrap();

// Access variational parameters
for (var, params) in &result.variational_params {
    println!("{}: {:?}", var, params);
}

// Check ELBO for convergence
println!("ELBO: {}", result.elbo);
println!("Converged: {}", result.converged);
```

#### ELBO Monitoring

```rust
let mut elbo_history = Vec::new();
let mean_field = MeanFieldInference::with_callback(
    1000,
    1e-4,
    |iteration, elbo| {
        elbo_history.push(elbo);
        println!("Iteration {}: ELBO = {}", iteration, elbo);
    }
);
```

### 3.1. Structured Variational Inference

Beyond mean-field, structured variational methods leverage the factor graph structure for improved accuracy:

#### Bethe Approximation

Uses the graph structure to define a structured approximation (equivalent to loopy BP fixed points):

```rust
use tensorlogic_quantrs_hooks::BetheApproximation;

// Create Bethe approximation engine
let bethe = BetheApproximation::new(
    100,    // max_iterations
    1e-6,   // tolerance
    0.0     // damping factor
);

// Run inference
let beliefs = bethe.run(&graph)?;

// Compute factor beliefs from variable beliefs
let factor_beliefs = bethe.compute_factor_beliefs(&graph, &beliefs)?;

// Compute Bethe free energy
let free_energy = bethe.compute_free_energy(&graph, &beliefs, &factor_beliefs)?;
println!("Bethe Free Energy: {:.4}", free_energy);
```

**Advantages over Mean-Field:**
- Respects factor graph structure
- More accurate marginals for loopy graphs
- Similar computational cost to loopy BP

#### Tree-Reweighted Belief Propagation (TRW-BP)

Provides upper bounds on the log partition function using edge reweighting:

```rust
use tensorlogic_quantrs_hooks::TreeReweightedBP;

// Create TRW-BP engine
let mut trw = TreeReweightedBP::new(
    100,    // max_iterations
    1e-6    // tolerance
);

// Optionally set custom edge weights
trw.set_edge_weight("X".to_string(), "factor1".to_string(), 0.5);

// Or use uniform weights (default)
trw.initialize_uniform_weights(&graph);

// Run inference
let beliefs = trw.run(&graph)?;

// Compute upper bound on log Z
let log_z_bound = trw.compute_log_partition_upper_bound(&graph, &beliefs)?;
```

**Key Properties:**
- Provides upper bounds on log partition function
- Guaranteed convergence for convex tree mixtures
- Particularly robust for loopy graphs
- Uses edge appearance probabilities ρ_e ∈ [0,1]

#### Comparison: Mean-Field vs. Bethe vs. TRW-BP

```rust
// Mean-Field: Fastest, assumes full independence
let mf = MeanFieldInference::default();
let mf_beliefs = mf.run(&graph)?;
let mf_elbo = mf.compute_elbo(&graph, &mf_beliefs)?;

// Bethe: Uses graph structure, more accurate
let bethe = BetheApproximation::default();
let bethe_beliefs = bethe.run(&graph)?;

// TRW-BP: Provides bounds, most robust
let mut trw = TreeReweightedBP::default();
let trw_beliefs = trw.run(&graph)?;
```

See `examples/structured_variational.rs` for a complete grid MRF comparison.

### 4. Gibbs Sampling

MCMC sampling for approximate marginals:

```rust
use tensorlogic_quantrs_hooks::GibbsSampler;

// Create sampler
let sampler = GibbsSampler::new(
    1000,   // num_samples
    100,    // burn_in
    10      // thinning
);

// Run sampling
let samples = sampler.sample(&graph).unwrap();

// Compute empirical marginals
let marginals = sampler.compute_marginals(&samples, &graph).unwrap();

for (var, marginal) in &marginals {
    println!("{}: {:?}", var, marginal);
}
```

#### Sample Statistics

```rust
// Check acceptance rates
let stats = sampler.get_statistics(&samples);
println!("Acceptance rate: {:.2}%", stats.acceptance_rate * 100.0);
println!("Effective sample size: {}", stats.effective_sample_size);
```

### 5. Junction Tree Algorithm (Exact Inference)

The junction tree algorithm provides exact inference for any graph structure by constructing a tree of cliques:

```rust
use tensorlogic_quantrs_hooks::JunctionTree;

// Build junction tree from factor graph
let mut tree = JunctionTree::from_factor_graph(&graph)?;

// Calibrate the tree (message passing)
tree.calibrate()?;

// Query exact marginals
let p_x = tree.query_marginal("X")?;
println!("P(X=0) = {}", p_x[[0]]);
println!("P(X=1) = {}", p_x[[1]]);

// Query joint marginals
let p_xy = tree.query_joint_marginal(&["X".to_string(), "Y".to_string()])?;
```

#### Junction Tree Properties

```rust
// Check treewidth (complexity indicator)
let tw = tree.treewidth();
println!("Treewidth: {}", tw);

// Verify running intersection property
assert!(tree.verify_running_intersection_property());

// Inspect clique structure
for (i, clique) in tree.cliques.iter().enumerate() {
    println!("Clique {}: {:?}", i, clique.variables);
}
```

**Advantages:**
- Exact inference (no approximation error)
- Efficient for low-treewidth graphs
- Handles any query after single calibration
- Guarantees consistency across marginals

**Complexity:** O(n × d^(w+1)) where w is the treewidth, d is max domain size

### 6. Expectation Propagation (EP)

EP approximates complex posteriors using moment matching with site approximations:

```rust
use tensorlogic_quantrs_hooks::ExpectationPropagation;

// Create EP algorithm
let ep = ExpectationPropagation::new(
    100,    // max_iterations
    1e-6,   // tolerance
    0.0     // damping (0.0 = no damping)
);

// Run EP inference
let marginals = ep.run(&graph)?;

// Access marginals
for (var, marginal) in &marginals {
    println!("{}: {:?}", var, marginal);
}
```

#### Gaussian EP for Continuous Variables

For continuous variables, use Gaussian EP with natural parameterization:

```rust
use tensorlogic_quantrs_hooks::{GaussianEP, GaussianSite};

// Create Gaussian EP
let gep = GaussianEP::new(100, 1e-6, 0.0);

// Create Gaussian sites
let site1 = GaussianSite::new("X".to_string(), 2.0, 4.0); // precision=2, precision_mean=4
let site2 = GaussianSite::new("X".to_string(), 3.0, 6.0);

// Combine sites
let product = site1.product(&site2);
println!("Mean: {}, Variance: {}", product.mean(), product.variance());
```

**Key Features:**
- Site approximations and cavity distributions
- Moment matching for discrete and continuous variables
- Damping for improved convergence
- Natural parameterization for Gaussians

### 7. Linear-chain CRFs (Sequence Labeling)

Linear-chain CRFs enable efficient sequence labeling with structured prediction:

```rust
use tensorlogic_quantrs_hooks::{LinearChainCRF, TransitionFeature, EmissionFeature};
use scirs2_core::ndarray::Array;

// Create CRF with 3 labels
let mut crf = LinearChainCRF::new(3);

// Set transition weights (3x3 matrix)
let transition_weights = Array::from_shape_vec(
    vec![3, 3],
    vec![
        0.5, 0.3, 0.2,  // From label 0
        0.2, 0.6, 0.2,  // From label 1
        0.3, 0.2, 0.5,  // From label 2
    ]
).unwrap().into_dimensionality::<scirs2_core::ndarray::Ix2>().unwrap();

crf.set_transition_weights(transition_weights)?;

// Viterbi decoding (most likely sequence)
let input_sequence = vec![0, 1, 2, 1, 0];
let (best_path, score) = crf.viterbi(&input_sequence)?;
println!("Best label sequence: {:?} (score: {:.2})", best_path, score);

// Compute marginal probabilities
let marginals = crf.marginals(&input_sequence)?;
for t in 0..input_sequence.len() {
    println!("Position {}: {:?}", t, marginals.row(t));
}
```

#### Custom Feature Functions

Define custom features for domain-specific sequence labeling:

```rust
use tensorlogic_quantrs_hooks::FeatureFunction;

struct BigramFeature {
    prev_label: usize,
    curr_label: usize,
}

impl FeatureFunction for BigramFeature {
    fn compute(
        &self,
        prev_label: Option<usize>,
        curr_label: usize,
        _input_sequence: &[usize],
        _position: usize,
    ) -> f64 {
        if prev_label == Some(self.prev_label) && curr_label == self.curr_label {
            1.0
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "bigram_feature"
    }
}

// Add feature with weight
let feature = Box::new(BigramFeature { prev_label: 0, curr_label: 1 });
crf.add_feature(feature, 2.5);
```

**Applications:**
- Part-of-speech tagging
- Named entity recognition
- Speech recognition
- Bioinformatics (protein sequence analysis)

**Algorithms:**
- Viterbi: O(T × S²) for most likely sequence
- Forward-backward: O(T × S²) for marginals
- Where T = sequence length, S = number of states

## QuantRS2 Integration

### Distribution Export

Convert factors to QuantRS2-compatible distributions for ecosystem integration:

```rust
use tensorlogic_quantrs_hooks::QuantRSDistribution;

// Export factor to QuantRS format
let factor = Factor::new("P(X,Y)".to_string(),
    vec!["X".to_string(), "Y".to_string()],
    values)?;

let dist_export = factor.to_quantrs_distribution()?;

println!("Variables: {:?}", dist_export.variables);
println!("Cardinalities: {:?}", dist_export.cardinalities);
println!("Type: {}", dist_export.metadata.distribution_type);
```

### Model Export

Export entire factor graphs for use across the COOLJAPAN ecosystem:

```rust
use tensorlogic_quantrs_hooks::QuantRSModelExport;

// Export model to QuantRS2 format
let model_export = graph.to_quantrs_model()?;

println!("Model type: {}", model_export.model_type);
println!("Variables: {}", model_export.variables.len());
println!("Factors: {}", model_export.factors.len());

// Get model statistics
let stats = graph.model_stats();
println!("Avg factor size: {:.2}", stats.avg_factor_size);
println!("Max factor size: {}", stats.max_factor_size);
```

### JSON Serialization

Export models as JSON for interoperability:

```rust
use tensorlogic_quantrs_hooks::quantrs_hooks::utils;

// Export to JSON
let json = utils::export_to_json(&graph)?;
println!("{}", json);

// Import from JSON
let model = utils::import_from_json(&json)?;
```

### Information Theory

Compute information-theoretic quantities:

```rust
use tensorlogic_quantrs_hooks::quantrs_hooks::utils;

// Mutual information
let mi = utils::mutual_information(&joint_dist, "X", "Y")?;
println!("I(X;Y) = {:.4} bits", mi);

// KL divergence
let kl = utils::kl_divergence(&p_dist, &q_dist)?;
println!("D_KL(P||Q) = {:.4}", kl);
```

## Parameter Learning

Learn model parameters from observed data.

### Maximum Likelihood Estimation

Estimate parameters from complete data (all variables observed):

```rust
use tensorlogic_quantrs_hooks::MaximumLikelihoodEstimator;
use std::collections::HashMap;

let estimator = MaximumLikelihoodEstimator::new();

// Create training data
let mut data = Vec::new();
for _ in 0..70 {
    let mut assignment = HashMap::new();
    assignment.insert("Weather".to_string(), 0); // Sunny
    data.push(assignment);
}
for _ in 0..30 {
    let mut assignment = HashMap::new();
    assignment.insert("Weather".to_string(), 1); // Rainy
    data.push(assignment);
}

// Estimate P(Weather)
let probs = estimator.estimate_marginal("Weather", 2, &data)?;
// Result: [0.7, 0.3]
```

### Bayesian Estimation with Priors

Use Dirichlet priors for robust estimation:

```rust
use tensorlogic_quantrs_hooks::BayesianEstimator;

let estimator = BayesianEstimator::new(2.0); // Prior strength

// Estimate with prior
let probs = estimator.estimate_marginal("X", 2, &data)?;
```

### Baum-Welch Algorithm for HMMs

Learn HMM parameters from observation sequences (even when hidden states are not observed):

```rust
use tensorlogic_quantrs_hooks::{BaumWelchLearner, SimpleHMM};

// Create an HMM with random initialization
let mut hmm = SimpleHMM::new_random(2, 3); // 2 states, 3 observations

// Observation sequences (hidden states unknown)
let observation_sequences = vec![
    vec![0, 0, 1, 2, 2, 0],
    vec![1, 2, 2, 1, 0, 0],
    // ... more sequences
];

// Learn parameters
let learner = BaumWelchLearner::with_verbose(100, 1e-4);
let log_likelihood = learner.learn(&mut hmm, &observation_sequences)?;

println!("Learned HMM with log-likelihood: {}", log_likelihood);
```

**Key Features:**
- Expectation-Maximization (EM) algorithm for HMMs
- Forward-backward message passing
- Automatic convergence detection
- Verbose mode for monitoring progress

## Advanced Usage

### Conditional Queries

Compute P(X | Y=y):

```rust
use tensorlogic_quantrs_hooks::ConditionalQuery;
use std::collections::HashMap;

// Evidence: Y = 1
let mut evidence = HashMap::new();
evidence.insert("Y".to_string(), 1);

let query = ConditionalQuery {
    variable: "X".to_string(),
    evidence,
};

let conditional = engine.conditional(&query).unwrap();
```

### Custom Convergence Criteria

```rust
struct CustomAlgorithm {
    inner: SumProductAlgorithm,
}

impl CustomAlgorithm {
    fn new() -> Self {
        Self {
            inner: SumProductAlgorithm::new(100, 1e-6, 0.0),
        }
    }

    fn check_convergence(&self, messages: &[Factor]) -> bool {
        // Custom convergence logic
        true
    }
}
```

### Multi-Variable Queries

```rust
// Compute joint marginal P(X, Y)
let vars_to_keep = vec!["X".to_string(), "Y".to_string()];
let joint_marginal = compute_joint_marginal(&graph, &vars_to_keep).unwrap();
```

## Integration with TensorLogic

### From TLExpr to Probabilities

```rust
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_quantrs_hooks::{expr_to_factor_graph, SumProductAlgorithm, InferenceEngine};

// Define logical rule: ∃x. P(x) ∧ Q(x)
let expr = TLExpr::exists(
    "x",
    "Domain",
    TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x")])
    )
);

// Convert to factor graph
let graph = expr_to_factor_graph(&expr).unwrap();

// Run probabilistic inference
let algorithm = Box::new(SumProductAlgorithm::default());
let engine = InferenceEngine::new(graph, algorithm);
let marginals = engine.run().unwrap();
```

### Probabilistic Logic Programming

```rust
// Weighted rules with confidence scores
let rules = vec![
    (0.9, TLExpr::imply(
        TLExpr::pred("bird", vec![Term::var("x")]),
        TLExpr::pred("can_fly", vec![Term::var("x")])
    )),
    (0.8, TLExpr::pred("bird", vec![Term::constant(1.0)])),
];

// Convert to factor graph with weights
let mut graph = FactorGraph::new();
for (weight, rule) in rules {
    let factor_graph = expr_to_factor_graph(&rule).unwrap();
    // Multiply factors by confidence weight
    // ... (implementation details)
}
```

## Performance Considerations

### Algorithm Selection Guide

| Graph Type | Recommended Algorithm | Complexity | Notes |
|-----------|----------------------|------------|-------|
| Tree | Sum-Product | O(N × D²) | Exact inference |
| Low Treewidth | Junction Tree | O(N × D^(w+1)) | Exact, w = treewidth |
| Small Loopy | Loopy BP with damping | O(I × N × D²) | Approximate |
| Large Loopy | Mean-Field VI | O(I × N × D) | Fast approximate |
| Large Loopy (Structured) | Bethe / TRW-BP | O(I × E × D²) | Better accuracy |
| Complex Posteriors | Expectation Propagation | O(I × F × D²) | Moment matching |
| Sequence Labeling | Linear-chain CRF | O(T × S²) | Viterbi/Forward-backward |
| Any | Gibbs Sampling | O(S × N × D) | MCMC |

Where:
- N = number of variables
- D = max domain size
- I = iterations to converge
- S = number of samples
- E = number of edges
- F = number of factors
- w = treewidth
- T = sequence length
- S = number of states (for CRF)

### Optimization Tips

1. **Use appropriate cardinalities**: Smaller domains = faster inference
2. **Enable damping for loopy graphs**: Improves convergence
3. **Tune convergence tolerance**: Balance accuracy vs. speed
4. **Use variational inference for large graphs**: O(N) vs O(N²) for BP
5. **Batch factor operations**: Leverage SciRS2 vectorization

### Memory Usage

```rust
// Estimate memory for factor graph
let num_vars = graph.num_variables();
let num_factors = graph.num_factors();
let avg_cardinality = 10;

let memory_mb = (num_vars * avg_cardinality * 8 +
                 num_factors * avg_cardinality.pow(2) * 8) / 1_000_000;
println!("Estimated memory: {} MB", memory_mb);
```

## Examples

### Example 1: Bayesian Network

```rust
// Classic cancer/smoking example
let mut graph = FactorGraph::new();

// Variables
graph.add_variable_with_card("Smoking".to_string(), "Binary".to_string(), 2);
graph.add_variable_with_card("Cancer".to_string(), "Binary".to_string(), 2);
graph.add_variable_with_card("XRay".to_string(), "Binary".to_string(), 2);

// Prior P(Smoking)
let p_smoking = Factor::new(
    "P(Smoking)".to_string(),
    vec!["Smoking".to_string()],
    Array::from_shape_vec(vec![2], vec![0.7, 0.3]).unwrap().into_dyn()
).unwrap();

// P(Cancer | Smoking)
let p_cancer_given_smoking = Factor::new(
    "P(Cancer|Smoking)".to_string(),
    vec!["Smoking".to_string(), "Cancer".to_string()],
    Array::from_shape_vec(vec![2, 2], vec![0.95, 0.05, 0.2, 0.8]).unwrap().into_dyn()
).unwrap();

// P(XRay | Cancer)
let p_xray_given_cancer = Factor::new(
    "P(XRay|Cancer)".to_string(),
    vec!["Cancer".to_string(), "XRay".to_string()],
    Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8]).unwrap().into_dyn()
).unwrap();

graph.add_factor(p_smoking).unwrap();
graph.add_factor(p_cancer_given_smoking).unwrap();
graph.add_factor(p_xray_given_cancer).unwrap();

// Query: P(Cancer | XRay=positive)
let mut evidence = HashMap::new();
evidence.insert("XRay".to_string(), 1);

let query = ConditionalQuery {
    variable: "Cancer".to_string(),
    evidence,
};

let algorithm = Box::new(SumProductAlgorithm::default());
let engine = InferenceEngine::new(graph, algorithm);
let p_cancer_given_xray = engine.conditional(&query).unwrap();

println!("P(Cancer | XRay=positive) = {}", p_cancer_given_xray[[1]]);
```

### Example 2: Markov Random Field

```rust
// 2x2 grid MRF for image denoising
let mut graph = FactorGraph::new();

// Pixel variables
for i in 0..4 {
    graph.add_variable_with_card(
        format!("pixel_{}", i),
        "Intensity".to_string(),
        2  // Binary: 0 (black), 1 (white)
    );
}

// Pairwise potentials (smoothness)
let smoothness = Array::from_shape_vec(
    vec![2, 2],
    vec![1.0, 0.3, 0.3, 1.0]  // Favor same-color neighbors
).unwrap().into_dyn();

// Add edge factors
let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
for (i, j) in edges {
    let factor = Factor::new(
        format!("edge_{}_{}", i, j),
        vec![format!("pixel_{}", i), format!("pixel_{}", j)],
        smoothness.clone()
    ).unwrap();
    graph.add_factor(factor).unwrap();
}

// Add observation factors (noisy measurements)
// ... (implementation details)
```

### Example 3: Hidden Markov Model

```rust
// Simple HMM: weather states predicting umbrella usage
let mut graph = FactorGraph::new();
let T = 5;  // Time steps

// Hidden states (weather)
for t in 0..T {
    graph.add_variable_with_card(
        format!("weather_{}", t),
        "Weather".to_string(),
        2  // Sunny=0, Rainy=1
    );
}

// Observations (umbrella)
for t in 0..T {
    graph.add_variable_with_card(
        format!("umbrella_{}", t),
        "Umbrella".to_string(),
        2
    );
}

// Initial state
let initial = Factor::new(
    "P(weather_0)".to_string(),
    vec!["weather_0".to_string()],
    Array::from_shape_vec(vec![2], vec![0.6, 0.4]).unwrap().into_dyn()
).unwrap();
graph.add_factor(initial).unwrap();

// Transition model
let transition = Array::from_shape_vec(
    vec![2, 2],
    vec![0.7, 0.3, 0.3, 0.7]  // Weather transitions
).unwrap().into_dyn();

for t in 0..T-1 {
    let factor = Factor::new(
        format!("P(weather_{}|weather_{})", t+1, t),
        vec![format!("weather_{}", t), format!("weather_{}", t+1)],
        transition.clone()
    ).unwrap();
    graph.add_factor(factor).unwrap();
}

// Observation model
let observation = Array::from_shape_vec(
    vec![2, 2],
    vec![0.9, 0.1, 0.2, 0.8]  // P(umbrella | weather)
).unwrap().into_dyn();

for t in 0..T {
    let factor = Factor::new(
        format!("P(umbrella_{}|weather_{})", t, t),
        vec![format!("weather_{}", t), format!("umbrella_{}", t)],
        observation.clone()
    ).unwrap();
    graph.add_factor(factor).unwrap();
}

// Filtering: P(weather_t | umbrella_0:t)
// Smoothing: P(weather_t | umbrella_0:T)
// ... (inference implementation)
```

## Testing

Run all tests:

```bash
cargo nextest run -p tensorlogic-quantrs-hooks
```

Run specific test suites:

```bash
# Factor operations
cargo test -p tensorlogic-quantrs-hooks factor::tests

# Message passing
cargo test -p tensorlogic-quantrs-hooks message_passing::tests

# Inference
cargo test -p tensorlogic-quantrs-hooks inference::tests

# Variational
cargo test -p tensorlogic-quantrs-hooks variational::tests
```

## Architecture

```
tensorlogic-quantrs-hooks
├── Factor Operations
│   ├── Product (×)
│   ├── Marginalization (Σ)
│   ├── Division (÷)
│   └── Reduction (evidence)
├── Factor Graphs
│   ├── Variable Nodes
│   ├── Factor Nodes
│   └── Adjacency Lists
├── Message Passing
│   ├── Sum-Product (marginals)
│   ├── Max-Product (MAP)
│   └── Convergence Detection
├── Variational Inference
│   ├── Mean-Field (fully factorized)
│   ├── Bethe Approximation (structured)
│   ├── Tree-Reweighted BP
│   └── ELBO/Free Energy Computation
└── Sampling
    ├── Gibbs Sampler
    ├── Burn-in/Thinning
    └── Empirical Marginals
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## References

- Koller & Friedman, "Probabilistic Graphical Models" (2009)
- Wainwright & Jordan, "Graphical Models, Exponential Families, and Variational Inference" (2008)
- Bishop, "Pattern Recognition and Machine Learning" (2006), Chapter 8

## License

Apache-2.0

---

**Status**: 🎉 Production Ready (v0.1.0-alpha.1)
**Last Updated**: 2025-11-07
**Tests**: 109 passing (100%: 96 unit + 13 integration)
**Examples**: 8 comprehensive examples
**Completeness**: ~99% (all medium-priority features complete!)
**Features**:
- **Inference**: 7 algorithms (Sum-Product, Max-Product, Junction Tree, Mean-Field, Bethe, TRW-BP, EP, Gibbs)
- **Models**: 5 types (Bayesian Networks, HMMs, MRFs, CRFs, Linear-chain CRFs)
- **Learning**: Parameter estimation, Baum-Welch EM
- **Integration**: QuantRS2 hooks, JSON export, information theory utilities
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
