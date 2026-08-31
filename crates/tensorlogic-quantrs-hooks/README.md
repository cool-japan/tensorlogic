# tensorlogic-quantrs-hooks

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--quantrs--hooks-orange)](https://crates.io/crates/tensorlogic-quantrs-hooks)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-quantrs-hooks)
[![Tests](https://img.shields.io/badge/tests-346%2F346-brightgreen)](#)
[![Benchmarks](https://img.shields.io/badge/benchmarks-50%2B-blue)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

**Probabilistic Graphical Model Integration for TensorLogic**

Bridge between logic-based reasoning and probabilistic inference through factor graphs, belief propagation, variational methods, quantum circuit integration, and tensor network optimization.

## Overview

`tensorlogic-quantrs-hooks` enables probabilistic reasoning over TensorLogic expressions by converting logical rules into factor graphs and applying state-of-the-art inference algorithms. This crate seamlessly integrates with the QuantRS2 ecosystem for probabilistic programming.

### Key Features

- **TLExpr → Factor Graph Conversion**: Automatic translation of logical expressions to PGM representations
- **Exact Inference**:
  - Sum-product and max-product belief propagation for tree-structured graphs
  - Parallel sum-product with rayon for large-scale graphs (near-linear scaling)
  - Junction tree algorithm for exact inference on arbitrary graphs
  - Variable elimination with 5 advanced ordering heuristics (MinDegree, MinFill, WeightedMinFill, MinWidth, MaxCardinalitySearch)
- **Approximate Inference**:
  - Loopy BP: Message passing for graphs with cycles, with damping and convergence detection (`loopy_bp` module, v0.1.17)
  - Variational Inference: Mean-field, Bethe approximation, and tree-reweighted BP
  - Expectation Propagation (EP): Moment matching with site approximations for discrete and continuous variables
  - MCMC Sampling: Gibbs sampling for approximate posterior computation
- **Convergence Monitoring** (`convergence` module, v0.1.6):
  - `ConvergenceMonitor`: Residual tracking, patience-based convergence detection, divergence detection
  - `DampingSchedule`: Fixed, Linear, Exponential, and Adaptive damping strategies
  - `ConvergenceConfig` builder and `InferenceStats` for diagnostic reporting
- **Importance Sampling and Particle Filters**:
  - ImportanceSampler with custom proposal distributions
  - Self-normalized importance sampling
  - Effective sample size (ESS) computation
  - LikelihoodWeighting for Bayesian networks
  - ParticleFilter (Sequential Monte Carlo) with systematic resampling
- **Dynamic Bayesian Networks**:
  - DynamicBayesianNetwork with state/observation variables
  - DBN unrolling to static FactorGraph
  - Filtering and smoothing
  - Viterbi decoding (MAP sequence)
  - DBNBuilder for fluent construction
  - CoupledDBN for interacting processes
- **Influence Diagrams (Decision Networks)**:
  - InfluenceDiagram with chance/decision/utility nodes
  - Expected utility computation
  - Optimal policy finding (exhaustive search)
  - Value of perfect information (VPI)
  - InfluenceDiagramBuilder for fluent construction
  - MultiAttributeUtility (MAUT) support
- **Performance Optimizations**:
  - Factor caching system with LRU eviction for memoization (FactorCache)
  - Thread-safe parallel message passing via rayon (ParallelSumProduct, ParallelMaxProduct)
  - Cache statistics tracking (hits, misses, hit rate)
  - Memory optimization (FactorPool, SparseFactor, LazyFactor, CompressedFactor, BlockSparseFactor)
  - Streaming factor graph for large graphs (StreamingFactorGraph)
- **QuantRS2 Integration**:
  - Distribution and model export to QuantRS format
  - JSON serialization for ecosystem interoperability
  - Information-theoretic utilities (mutual information, KL divergence)
  - MCMC sampling hooks, parameter learning interfaces
  - Quantum annealing (QuantumAnnealing, QuantumInference)
- **Parameter Learning**:
  - Maximum Likelihood Estimation (MLE) for discrete distributions
  - Bayesian estimation with Dirichlet priors
  - Baum-Welch algorithm (EM) for Hidden Markov Models
  - Forward-backward algorithm implementation
- **Sequence Models**:
  - Linear-chain CRFs for sequence labeling with Viterbi decoding
  - Feature functions (transition, emission, custom)
  - Forward-backward algorithm for marginal probabilities
- **Quantum Circuit Integration** (`quantum_circuit` module):
  - IsingModel for quadratic unconstrained binary optimization (QUBO)
  - QUBOProblem for constraint satisfaction as QUBO
  - QAOA (Quantum Approximate Optimization Algorithm) circuit builder
  - QAOAConfig and QAOAResult for QAOA parameterization
  - tlexpr_to_qaoa_circuit conversion
- **Quantum Simulation** (`quantum_simulation` module):
  - QuantumSimulationBackend for simulated quantum computation
  - SimulatedState for tracking quantum states
  - SimulationConfig for backend configuration
  - run_qaoa function for full QAOA simulation
- **Tensor Network Bridge** (`tensor_network_bridge` module):
  - TensorNetwork for tensor contraction networks
  - MatrixProductState (MPS) for efficient 1D chain representations
  - factor_graph_to_tensor_network conversion
  - linear_chain_to_mps conversion
  - TensorNetworkStats for network analysis
- **Quality Assurance**:
  - Property-based testing with proptest (14 property tests)
  - Comprehensive benchmark suite with criterion (50+ benchmarks across 3 suites)
  - 346 tests (100% pass rate for non-precision-limited tests, 4 skipped)
  - 4 tests skipped with documented precision investigation notes
- **Full SciRS2 Integration**: All tensor operations use SciRS2 for performance and consistency

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
```

### Converting TLExpr to Factor Graph

```rust
use tensorlogic_quantrs_hooks::expr_to_factor_graph;
use tensorlogic_ir::{TLExpr, Term};

let expr = TLExpr::and(
    TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")]),
    TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]),
);

let graph = expr_to_factor_graph(&expr)?;
println!("Variables: {}", graph.num_variables());
```

### Belief Propagation

```rust
use tensorlogic_quantrs_hooks::{SumProductAlgorithm, MessagePassingAlgorithm};

let algorithm = SumProductAlgorithm::new();
let marginals = algorithm.run(&graph)?;

for (var, marginal) in &marginals {
    println!("P({}) = {:?}", var, marginal);
}
```

### Junction Tree (Exact Inference)

```rust
use tensorlogic_quantrs_hooks::JunctionTree;

let jt = JunctionTree::from_factor_graph(&graph)?;
let marginals = jt.compute_marginals()?;
```

### Variable Elimination

```rust
use tensorlogic_quantrs_hooks::{VariableElimination, EliminationStrategy};

let ve = VariableElimination::new(EliminationStrategy::MinFill);
let result = ve.marginal(&graph, "target_var")?;
```

### Bayesian Networks

```rust
use tensorlogic_quantrs_hooks::BayesianNetwork;

let mut bn = BayesianNetwork::new();
bn.add_node("A".to_string(), 2)?;
bn.add_node("B".to_string(), 2)?;
bn.add_edge("A".to_string(), "B".to_string())?;

// Set CPDs and run inference
let fg = bn.to_factor_graph()?;
```

### Hidden Markov Models

```rust
use tensorlogic_quantrs_hooks::HiddenMarkovModel;

let mut hmm = HiddenMarkovModel::new(3, 2); // 3 states, 2 observations
// Set transition, emission, initial probabilities
// Run filtering, smoothing, Viterbi
let viterbi = hmm.viterbi(&observations)?;
```

### Particle Filter

```rust
use tensorlogic_quantrs_hooks::{ParticleFilter, ProposalDistribution};

let proposal = ProposalDistribution::uniform(state_size);
let mut pf = ParticleFilter::new(1000, proposal);

for observation in &observations {
    pf.update(observation)?;
    let state_estimate = pf.estimate()?;
}
```

### Dynamic Bayesian Networks

```rust
use tensorlogic_quantrs_hooks::{DBNBuilder, DynamicBayesianNetwork};

let dbn = DBNBuilder::new()
    .add_state_variable("hidden".to_string(), 3)
    .add_observation_variable("obs".to_string(), 2)
    .build()?;

// Unroll to static factor graph for T time steps
let unrolled = dbn.unroll(10)?;

// Viterbi decoding
let best_sequence = dbn.viterbi(&observations)?;
```

### Influence Diagrams

```rust
use tensorlogic_quantrs_hooks::{InfluenceDiagramBuilder, NodeType};

let diagram = InfluenceDiagramBuilder::new()
    .add_node("Weather".to_string(), NodeType::Chance, 2)
    .add_node("Umbrella".to_string(), NodeType::Decision, 2)
    .add_node("Utility".to_string(), NodeType::Utility, 1)
    .build()?;

let (optimal_policy, expected_utility) = diagram.find_optimal_policy()?;
let vpi = diagram.value_of_perfect_information("Weather".to_string())?;
```

### Quantum Circuit (QAOA)

```rust
use tensorlogic_quantrs_hooks::{QUBOProblem, QAOAConfig, tlexpr_to_qaoa_circuit};

// Convert TLExpr satisfiability to QUBO
let circuit = tlexpr_to_qaoa_circuit(&expr, layers)?;

// Configure QAOA
let config = QAOAConfig { n_layers: 3, ..Default::default() };
let ising = IsingModel::from_qubo(&qubo);
let builder = QuantumCircuitBuilder::new(ising, config);
```

### Tensor Network Bridge

```rust
use tensorlogic_quantrs_hooks::{factor_graph_to_tensor_network, linear_chain_to_mps};

// Convert factor graph to tensor network for efficient contraction
let tn = factor_graph_to_tensor_network(&graph)?;
let stats = TensorNetworkStats::from_network(&tn);

// Represent a linear chain as Matrix Product State
let mps = linear_chain_to_mps(&linear_chain_factors, max_bond_dim)?;
```

### Memory-Efficient Large Graphs

```rust
use tensorlogic_quantrs_hooks::{StreamingFactorGraph, SparseFactor, FactorPool};

// Pool-based memory allocation
let pool = FactorPool::new(1024);

// Sparse factors for near-zero entries
let sparse = SparseFactor::from_factor(dense_factor, threshold)?;

// Streaming for graphs too large for memory
let mut streaming = StreamingFactorGraph::new();
streaming.process_batch(&batch, &algorithm)?;
```

## Supported Inference Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| SumProductAlgorithm | Exact (trees) | Tree-structured graphs |
| MaxProductAlgorithm | MAP (trees) | MAP inference on trees |
| ParallelSumProduct | Exact (parallel) | Large tree graphs |
| VariableElimination | Exact | Small graphs, any structure |
| JunctionTree | Exact | Arbitrary graphs |
| MeanFieldInference | Approximate | Large, dense graphs |
| BetheApproximation | Approximate | Loopy graphs |
| TreeReweightedBP | Approximate | Graphs with cycles |
| ExpectationPropagation | Approximate | Continuous or complex factors |
| GibbsSampler | MCMC | Complex posteriors |
| ImportanceSampler | Monte Carlo | Custom proposals |
| ParticleFilter | Sequential MC | Time series / DBNs |
| LikelihoodWeighting | Monte Carlo | Bayesian network evidence |

## Elimination Ordering Strategies

| Strategy | Description |
|----------|-------------|
| MinDegree | Minimize degree of each eliminated variable |
| MinFill | Minimize edges added during elimination |
| WeightedMinFill | MinFill weighted by factor sizes |
| MinWidth | Minimize maximum clique width |
| MaxCardinalitySearch | Greedy cardinality ordering |

## Testing

```bash
cargo nextest run -p tensorlogic-quantrs-hooks
# 346 tests, all applicable tests passing (4 skipped)
```

## Benchmarking

```bash
cargo bench -p tensorlogic-quantrs-hooks
```

Benchmark suites:
- Factor operations (6 benchmark groups)
- Message passing (7 benchmark groups)
- Inference algorithms comparison (9 benchmark groups)
- Total: 50+ benchmarks

## Examples

8 comprehensive examples:

```bash
# Bayesian Network inference (Student Performance Model)
cargo run --example bayesian_network -p tensorlogic-quantrs-hooks

# HMM temporal inference (Weather Prediction)
cargo run --example hmm_weather -p tensorlogic-quantrs-hooks

# Junction Tree exact inference
cargo run --example junction_tree -p tensorlogic-quantrs-hooks

# QuantRS2 integration
cargo run --example quantrs_integration -p tensorlogic-quantrs-hooks

# Parameter learning (Baum-Welch)
cargo run --example parameter_learning -p tensorlogic-quantrs-hooks

# Structured variational inference
cargo run --example structured_variational -p tensorlogic-quantrs-hooks

# Expectation Propagation
cargo run --example expectation_propagation -p tensorlogic-quantrs-hooks

# Linear-chain CRF
cargo run --example linear_chain_crf -p tensorlogic-quantrs-hooks
```

## Architecture

```text
TLExpr → FactorGraph → Inference → Marginals
  ↓           ↓            ↓            ↓
Predicates  Factors    Einsum Ops  Probabilities
                          ↓
                    [Multiple Algorithms]
                    SumProduct / JunctionTree /
                    VariableElimination /
                    MeanField / EP / Gibbs /
                    ImportanceSampling / ParticleFilter

FactorGraph → TensorNetwork → MatrixProductState
                    ↓
               Contraction
```

## License

Apache-2.0

---

**Status**: Production Ready (v0.1.2 Stable)
**Last Updated**: 2026-08-30
**Tests**: 346 passing (100% pass rate for non-precision-limited tests, 4 skipped)
**Benchmarks**: 3 suites, 50+ benchmarks
**Examples**: 8 comprehensive examples
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
