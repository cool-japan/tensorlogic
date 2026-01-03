# Getting Started with TensorLogic

**A practical guide to building neural-symbolic AI systems with TensorLogic**

This guide will walk you through the TensorLogic ecosystem, from basic concepts to advanced applications, with hands-on examples at each step.

## Table of Contents

1. [What is TensorLogic?](#what-is-tensorlogic)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Your First Logic-to-Tensor Program](#your-first-logic-to-tensor-program)
5. [Working with the Ecosystem](#working-with-the-ecosystem)
6. [Advanced Topics](#advanced-topics)
7. [Real-World Examples](#real-world-examples)
8. [Troubleshooting](#troubleshooting)

---

## What is TensorLogic?

TensorLogic is a **logic-as-tensor compilation framework** that bridges symbolic reasoning and neural computation. It:

- **Compiles logical rules** (∀x. P(x) ∧ Q(x) → R(x)) into **tensor operations** (einsum graphs)
- **Enables differentiable reasoning** through automatic differentiation
- **Integrates seamlessly** with neural networks, probabilistic models, and knowledge graphs
- **Provides multiple backends** (SciRS2 CPU/SIMD, future GPU support)

### When to Use TensorLogic

✅ **Perfect for:**
- Combining symbolic knowledge with neural learning
- Enforcing logical constraints during training
- Probabilistic logic programming with PGMs
- Knowledge graph reasoning with tensor operations
- Neurosymbolic AI research and applications

❌ **Not ideal for:**
- Pure deep learning (use PyTorch/TensorFlow directly)
- Traditional logic programming (use Prolog/ASP)
- Simple rule-based systems (use production rule engines)

---

## Installation

### Prerequisites

- **Rust**: 1.70+ (for Rust development)
- **Python**: 3.9+ (for Python bindings)
- **SciRS2**: Automatically included via workspace dependencies

### Rust Installation

```bash
# Clone the repository
git clone https://github.com/cool-japan/tensorlogic.git
cd tensorlogic

# Build all crates
cargo build --workspace --release

# Run tests to verify installation
cargo test --workspace --lib

# Expected output: 1,225+ tests passing
```

### Python Installation

```bash
# Install from source (development)
cd crates/tensorlogic-py
maturin develop --release

# Verify installation
python -c "import tensorlogic_py as tl; print(tl.__version__)"
```

---

## Core Concepts

### 1. The TensorLogic Stack

```
┌─────────────────────────────────────┐
│   Applications & Integration        │
│  (OxiRS, SkleaRS, QuantrS2, etc.)  │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Training & Inference           │
│  (tensorlogic-train, -infer)        │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       Execution Backend             │
│    (tensorlogic-scirs-backend)      │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Compiler & Optimizer           │
│   (tensorlogic-compiler)            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      IR & Core Types                │
│      (tensorlogic-ir)               │
└─────────────────────────────────────┘
```

### 2. Key Abstractions

**TLExpr** (Logical Expression):
- Represents logical formulas: predicates, quantifiers, connectives
- Example: `∀x. Bird(x) → CanFly(x)`

**EinsumGraph** (Tensor Graph):
- Compiled tensor computation graph
- Nodes represent einsum operations
- Edges represent data flow

**Executor**:
- Runs EinsumGraphs on specific backends
- Supports forward pass (inference) and backward pass (training)

---

## Your First Logic-to-Tensor Program

### Step 1: Define Logical Rules

```rust
use tensorlogic_ir::{TLExpr, Term};

// Define variables
let x = Term::var("x");
let y = Term::var("y");

// Logical rule: friends(x, y) ∧ likes(y, z) → likes(x, z)
let friends = TLExpr::pred("friends", vec![x.clone(), y.clone()]);
let likes_yz = TLExpr::pred("likes", vec![y.clone(), Term::var("z")]);

let rule = TLExpr::imply(
    TLExpr::and(friends, likes_yz),
    TLExpr::pred("likes", vec![x, Term::var("z")])
);
```

### Step 2: Compile to Tensor Graph

```rust
use tensorlogic_compiler::TlCompiler;
use tensorlogic_adapters::SymbolTable;

// Set up symbol table
let mut symbols = SymbolTable::new();
symbols.add_predicate("friends", vec!["Person", "Person"]);
symbols.add_predicate("likes", vec!["Person", "Thing"]);
symbols.add_domain("Person", 100);  // 100 people
symbols.add_domain("Thing", 50);    // 50 things

// Compile to einsum graph
let compiler = TlCompiler::new(symbols);
let graph = compiler.compile(&rule)?;

println!("Compiled to {} nodes", graph.nodes.len());
```

### Step 3: Execute with Data

```rust
use tensorlogic_scirs_backend::Scirs2Exec;
use tensorlogic_infer::TlExecutor;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// Create executor
let executor = Scirs2Exec::new();

// Prepare tensor data
let mut inputs = HashMap::new();
inputs.insert("friends".to_string(), Array2::zeros((100, 100)));
inputs.insert("likes".to_string(), Array2::zeros((100, 50)));

// Execute
let outputs = executor.execute(&graph, &inputs)?;

println!("Result shape: {:?}", outputs[0].shape());
```

---

## Working with the Ecosystem

### Training with tensorlogic-train

```rust
use tensorlogic_train::{
    Trainer, TrainerConfig, AdamOptimizer, OptimizerConfig,
    LogicalLoss, LossConfig, CrossEntropyLoss,
    RuleSatisfactionLoss,
};

// Configure logical loss
let loss_config = LossConfig {
    supervised_weight: 1.0,
    rule_weight: 10.0,  // Heavily enforce rules
    ..Default::default()
};

let loss = LogicalLoss::new(
    loss_config,
    Box::new(CrossEntropyLoss::default()),
    vec![Box::new(RuleSatisfactionLoss::default())],
    vec![],
);

// Configure optimizer with gradient clipping
let optimizer = AdamOptimizer::new(OptimizerConfig {
    learning_rate: 0.001,
    grad_clip: Some(5.0),
    grad_clip_mode: GradClipMode::Norm,
    ..Default::default()
});

// Create trainer
let config = TrainerConfig {
    num_epochs: 100,
    batch_size: 32,
    ..Default::default()
};

let trainer = Trainer::new(config, Box::new(loss), Box::new(optimizer));

// Train with data
let history = trainer.train(
    &train_data.view(),
    &train_targets.view(),
    Some(&val_data.view()),
    Some(&val_targets.view()),
    &mut parameters,
)?;
```

### Probabilistic Inference with tensorlogic-quantrs-hooks

```rust
use tensorlogic_quantrs_hooks::{
    FactorGraph, Factor, SumProductAlgorithm, InferenceEngine,
};
use scirs2_core::ndarray::Array;

// Create factor graph
let mut graph = FactorGraph::new();

// Add variables
graph.add_variable_with_card("Rain".to_string(), "Boolean".to_string(), 2);
graph.add_variable_with_card("Sprinkler".to_string(), "Boolean".to_string(), 2);

// Add factors (CPTs)
let p_rain = Factor::new(
    "P(Rain)".to_string(),
    vec!["Rain".to_string()],
    Array::from_shape_vec(vec![2], vec![0.3, 0.7]).unwrap().into_dyn()
).unwrap();

graph.add_factor(p_rain).unwrap();

// Run belief propagation
let algorithm = Box::new(SumProductAlgorithm::default());
let engine = InferenceEngine::new(graph, algorithm);

let marginals = engine.run()?;
println!("P(Rain) = {:?}", marginals.get("Rain"));
```

### Knowledge Graph Integration with tensorlogic-oxirs-bridge

```rust
use tensorlogic_oxirs_bridge::{RdfToTlConverter, ShaclValidator};

// Load RDF knowledge graph
let rdf_data = r#"
@prefix : <http://example.org/> .
:Alice :knows :Bob .
:Bob :knows :Charlie .
"#;

// Convert RDF to TensorLogic expressions
let converter = RdfToTlConverter::new();
let tl_exprs = converter.convert_rdf_to_tl(rdf_data)?;

// Validate with SHACL shapes
let shacl_shape = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
:PersonShape a sh:NodeShape ;
    sh:targetClass :Person ;
    sh:property [
        sh:path :knows ;
        sh:minCount 1 ;
    ] .
"#;

let validator = ShaclValidator::new();
let violations = validator.validate(rdf_data, shacl_shape)?;
```

---

## Advanced Topics

### 1. Custom Loss Functions

```rust
use tensorlogic_train::{Loss, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

#[derive(Debug, Clone)]
pub struct CustomConstraintLoss {
    pub penalty_weight: f64,
}

impl Loss for CustomConstraintLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        // Implement custom logic
        let mut total = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let violation = (predictions[[i, j]] - targets[[i, j]]).abs();
                if violation > 0.1 {
                    total += self.penalty_weight * violation.powi(2);
                }
            }
        }
        Ok(total / predictions.nrows() as f64)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        // Compute gradients...
        Ok(grad)
    }
}
```

### 2. Streaming Execution for Large Datasets

```rust
use tensorlogic_infer::{TlStreamingExecutor, StreamingConfig, StreamingMode};

let config = StreamingConfig::new(StreamingMode::Adaptive {
    initial_chunk: 64,
})
.with_prefetch(2)
.with_checkpointing(100);

let results = executor.execute_stream(&graph, input_stream, &config)?;

for result in results {
    println!("Processed chunk {} in {}ms",
        result.metadata.chunk_id,
        result.processing_time_ms
    );
}
```

### 3. Multi-Device Placement

```rust
use tensorlogic_infer::{PlacementOptimizer, PlacementStrategy, Device};

let devices = vec![Device::CPU(0), Device::GPU(0)];
let optimizer = PlacementOptimizer::new(devices, PlacementStrategy::LoadBalance);

let plan = optimizer.optimize(&graph)?;

println!("Placement plan:");
for (node_id, device) in &plan.node_placements {
    println!("  Node {} → {:?}", node_id, device);
}
```

### 4. Regularization and Data Augmentation

```rust
use tensorlogic_train::{
    L2Regularization, Regularizer,
    NoiseAugmenter, MixupAugmenter, CompositeAugmenter, DataAugmenter,
};

// Regularization
let regularizer = L2Regularization::new(0.01);
let penalty = regularizer.compute_penalty(&parameters)?;

// Data augmentation pipeline
let mut augmenter = CompositeAugmenter::new();
augmenter.add(Box::new(NoiseAugmenter::new(0.0, 0.05)));
augmenter.add(Box::new(MixupAugmenter::new(1.0)));

let augmented = augmenter.augment(&data.view())?;
```

---

## Real-World Examples

### Example 1: Knowledge Base Completion

**Problem**: Predict missing relationships in a knowledge graph.

```rust
use tensorlogic_ir::TLExpr;
use tensorlogic_compiler::TlCompiler;
use tensorlogic_train::{Trainer, TrainerConfig, ContrastiveLoss};

// Rule: Similar entities have similar relationships
// ∀x,y,z. Similar(x,y) ∧ HasRel(y,z) → HasRel(x,z)

let similar = TLExpr::pred("Similar", vec![Term::var("x"), Term::var("y")]);
let has_rel_yz = TLExpr::pred("HasRel", vec![Term::var("y"), Term::var("z")]);
let has_rel_xz = TLExpr::pred("HasRel", vec![Term::var("x"), Term::var("z")]);

let rule = TLExpr::imply(
    TLExpr::and(similar, has_rel_yz),
    has_rel_xz
);

// Compile and train with contrastive loss for embeddings
let graph = compile(&rule)?;
let loss = ContrastiveLoss::new(1.0);

// Train to learn embeddings that satisfy the rule
// ...
```

### Example 2: Constraint-Aware Classification

**Problem**: Classify images with logical constraints (e.g., "if it has wings, it's likely a bird").

```rust
use tensorlogic_train::{LogicalLoss, ConstraintViolationLoss};

// Define constraint: HasWings(x) → IsBird(x) with high confidence
let has_wings = TLExpr::pred("HasWings", vec![Term::var("x")]);
let is_bird = TLExpr::pred("IsBird", vec![Term::var("x")]);
let constraint = TLExpr::imply(has_wings, is_bird);

// Use logical loss to enforce constraint during training
let logical_loss = LogicalLoss::new(
    LossConfig {
        supervised_weight: 1.0,
        constraint_weight: 5.0,  // High penalty for violations
        ..Default::default()
    },
    Box::new(CrossEntropyLoss::default()),
    vec![],
    vec![Box::new(ConstraintViolationLoss::default())],
);

// Train neural network with constraint enforcement
// ...
```

### Example 3: Probabilistic Question Answering

**Problem**: Answer queries over uncertain knowledge using probabilistic inference.

```rust
use tensorlogic_quantrs_hooks::{FactorGraph, SumProductAlgorithm};

// Knowledge: "Bob is friends with Alice (0.9 probability)"
//           "Alice likes pizza (0.8 probability)"
//           "Friends usually share food preferences (0.7)"

// Build factor graph
let mut graph = FactorGraph::new();
// ... add variables and factors ...

// Query: P(Bob likes pizza | evidence)
let algorithm = SumProductAlgorithm::new(100, 1e-6, 0.0);
let engine = InferenceEngine::new(graph, Box::new(algorithm));

let marginal = engine.marginalize(&MarginalizationQuery {
    variable: "BobLikesPizza".to_string(),
})?;

println!("P(Bob likes pizza) = {:.3}", marginal[[1]]);
```

---

## Troubleshooting

### Common Issues

#### 1. Compilation Errors: "Type mismatch"

**Problem**: Domain mismatches in predicates.

```rust
// ❌ Wrong: Inconsistent domains
symbols.add_predicate("knows", vec!["Person", "Thing"]);
let expr = TLExpr::pred("knows", vec![person_var, person_var2]);

// ✅ Correct: Matching domains
symbols.add_predicate("knows", vec!["Person", "Person"]);
```

#### 2. Runtime Errors: "Tensor shape mismatch"

**Problem**: Input tensors don't match expected dimensions.

```rust
// Check expected shapes from symbol table
println!("Expected shape for 'knows': ({}, {})",
    symbols.get_domain_size("Person")?,
    symbols.get_domain_size("Person")?
);

// Ensure input tensors match
let knows_tensor = Array2::zeros((
    symbols.get_domain_size("Person")?,
    symbols.get_domain_size("Person")?
));
```

#### 3. Performance: "Inference is slow"

**Solutions**:
```rust
// 1. Enable SIMD backend
cargo build --features simd

// 2. Use batch execution
let results = executor.execute_batch_parallel(&graph, batch_inputs, Some(4))?;

// 3. Enable memory pooling
let mut pool = MemoryPool::new();
// ... executor uses pool for allocations

// 4. Use streaming for large data
let config = StreamingConfig::new(StreamingMode::DynamicChunk {
    target_memory_mb: 512,
});
```

#### 4. Training: "Loss not decreasing"

**Debugging steps**:
```rust
// 1. Check learning rate
let lr_finder = LearningRateFinder::new(1e-7, 10.0, 100, true);
// ... run LR finder to find optimal LR

// 2. Monitor gradients
let grad_monitor = GradientMonitor::new(1e-5, 100.0);
callbacks.add(Box::new(grad_monitor));
// Check for vanishing/exploding gradients

// 3. Reduce constraint weight
let loss_config = LossConfig {
    supervised_weight: 1.0,
    constraint_weight: 0.1,  // Start small, increase gradually
    ..Default::default()
};

// 4. Use gradient clipping
let optimizer = AdamOptimizer::new(OptimizerConfig {
    grad_clip: Some(1.0),
    grad_clip_mode: GradClipMode::Norm,
    ..Default::default()
});
```

### Getting Help

- **Documentation**: Check crate-level README files in `crates/*/README.md`
- **Examples**: Browse `crates/*/examples/` for working code
- **Tests**: Look at `crates/*/src/**/*_tests.rs` for usage patterns
- **Issues**: Report bugs at https://github.com/cool-japan/tensorlogic/issues
- **Discussions**: Ask questions in GitHub Discussions

---

## Next Steps

### Beginner Path

1. ✅ Complete this guide
2. ⏭️ Run examples in `crates/tensorlogic-compiler/examples/`
3. ⏭️ Build a simple knowledge graph completion system
4. ⏭️ Explore training with constraints using `tensorlogic-train`

### Intermediate Path

1. ✅ Master basic compilation and execution
2. ⏭️ Study `tensorlogic-quantrs-hooks` for probabilistic reasoning
3. ⏭️ Implement custom loss functions
4. ⏭️ Integrate with existing neural architectures

### Advanced Path

1. ✅ Understand the full compilation pipeline
2. ⏭️ Contribute custom optimization passes
3. ⏭️ Implement domain-specific backends
4. ⏭️ Research novel neurosymbolic architectures

### Further Reading

- **[Architecture Overview](ARCHITECTURE.md)**: System design and internals
- **[Loss Function Guide](crates/tensorlogic-train/docs/LOSS_FUNCTIONS.md)**: Choosing the right loss
- **[Hyperparameter Tuning](crates/tensorlogic-train/docs/HYPERPARAMETER_TUNING.md)**: Optimization strategies
- **[SciRS2 Integration Policy](SCIRS2_INTEGRATION_POLICY.md)**: Backend integration guidelines

---

## Appendix: Cheat Sheet

### Quick Reference

```rust
// Import commonly used items
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_compiler::TlCompiler;
use tensorlogic_scirs_backend::Scirs2Exec;
use tensorlogic_infer::TlExecutor;
use tensorlogic_train::{Trainer, TrainerConfig, AdamOptimizer};

// Build logical expression
let expr = TLExpr::and(
    TLExpr::pred("P", vec![Term::var("x")]),
    TLExpr::pred("Q", vec![Term::var("x")])
);

// Compile
let graph = compiler.compile(&expr)?;

// Execute
let outputs = executor.execute(&graph, &inputs)?;

// Train
let history = trainer.train(&data, &targets, None, None, &mut params)?;
```

### Common Patterns

```rust
// Quantifiers
TLExpr::exists("x", "Domain", body)
TLExpr::forall("x", "Domain", body)

// Connectives
TLExpr::and(left, right)
TLExpr::or(left, right)
TLExpr::not(expr)
TLExpr::imply(premise, conclusion)

// Arithmetic
TLExpr::add(left, right)
TLExpr::mul(left, right)

// Aggregation
TLExpr::aggregate(AggregateOp::Sum, "x", "Domain", body, vec![])
```

---

**Happy Logic-as-Tensor Programming! 🚀**

For more examples and detailed documentation, explore the `crates/` directory and individual crate READMEs.

**Version**: 0.1.0-alpha.2
**Status**: Production Ready
