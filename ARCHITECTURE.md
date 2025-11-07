# TensorLogic Architecture

**Version:** 0.1.0

**Audience:** Developers, contributors, and advanced users

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Module Organization](#module-organization)
4. [Compilation Pipeline](#compilation-pipeline)
5. [Type System](#type-system)
6. [Execution Model](#execution-model)
7. [Memory Management](#memory-management)
8. [Integration Patterns](#integration-patterns)
9. [Performance Considerations](#performance-considerations)
10. [Extension Points](#extension-points)
11. [Design Decisions](#design-decisions)

---

## Overview

TensorLogic is a **logic-as-tensor compilation framework** that bridges symbolic reasoning and neural computation. The architecture is designed around three core principles:

1. **Separation of Concerns**: Planning layer (symbolic) vs. execution layer (numeric)
2. **Backend Agnostic**: Trait-based abstractions allow multiple execution engines
3. **Ecosystem Integration**: First-class interop with SciRS2, OxiRS, QuantrS2, SkleaRS, TrustformeRS

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input Layer                          │
│  (Rust DSL, Python API, RDF/SHACL, GraphQL)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Planning Layer (Engine-Agnostic)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ TensorLogic │→ │ TensorLogic │→ │ EinsumGraph │         │
│  │     IR      │  │  Compiler   │  │  (Output)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│     (AST/Terms)   (Optimization)    (Executable Plan)       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Execution Layer (Backend-Specific)              │
│  ┌──────────────────────────────────────────────────┐       │
│  │          TlExecutor Trait (Abstract)              │       │
│  └─────────┬────────────────────────────────────────┘       │
│            │                                                  │
│  ┌─────────▼─────────┐  ┌──────────────┐  ┌──────────┐     │
│  │ SciRS2 Backend    │  │ Custom ONNX  │  │  GPU     │     │
│  │ (CPU/SIMD/GPU)    │  │   Backend    │  │ Backend  │     │
│  └───────────────────┘  └──────────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Philosophy

### 1. **Compile Once, Execute Anywhere**

The planning layer produces a **backend-agnostic execution plan** (EinsumGraph). This allows:
- Static optimization (fusion, CSE, dead code elimination)
- Backend selection at runtime
- Portable serialization of compiled plans

### 2. **Explicit Over Implicit**

- Domain definitions are explicit (no automatic type inference)
- Axis assignments are traceable (SymbolTable → CompilerContext)
- Metadata propagates through all transformations

### 3. **Composability**

All components are designed for composition:
- Loss functions: `CompositeLoss`
- Regularizers: `CompositeRegularization`
- Augmenters: `CompositeAugmenter`
- Logging: `MetricsLogger` with multiple backends

### 4. **Zero-Cost Abstractions**

- Trait-based design with monomorphization
- Minimal runtime overhead (traits resolved at compile-time)
- Direct SciRS2 integration without wrapper layers

---

## Module Organization

The workspace is organized into **11 specialized crates** with clear dependency boundaries:

```
tensorlogic/
├── tensorlogic-ir              # Core AST and intermediate representation
├── tensorlogic-compiler        # Logic → Tensor compilation
├── tensorlogic-infer           # Execution traits (TlExecutor, TlAutodiff)
├── tensorlogic-adapters        # Symbol tables, domain management
│
├── tensorlogic-scirs-backend   # Reference SciRS2 executor
├── tensorlogic-train           # Training loops, losses, optimizers
│
├── tensorlogic-oxirs-bridge    # RDF/SHACL/GraphQL integration
├── tensorlogic-sklears-kernels # SkleaRS similarity kernels
├── tensorlogic-quantrs-hooks   # PGM/message-passing interop
├── tensorlogic-trustformers    # Transformer-as-rules bindings
└── tensorlogic-py              # Python bindings (PyO3)
```

### Dependency Graph

```
tensorlogic-py
    ├── tensorlogic-train
    │   ├── tensorlogic-infer
    │   └── tensorlogic-scirs-backend
    │       └── tensorlogic-infer
    ├── tensorlogic-compiler
    │   ├── tensorlogic-ir
    │   └── tensorlogic-adapters
    └── tensorlogic-oxirs-bridge
        ├── tensorlogic-compiler
        └── tensorlogic-adapters

tensorlogic-trustformers
    └── tensorlogic-compiler

tensorlogic-quantrs-hooks
    └── tensorlogic-ir

tensorlogic-sklears-kernels
    └── tensorlogic-compiler
```

**Key Design Decisions:**

1. **tensorlogic-ir** has zero dependencies (except serde) - it's the foundational type system
2. **tensorlogic-infer** defines traits but doesn't depend on SciRS2 - allows alternative backends
3. **tensorlogic-scirs-backend** is the only crate that directly depends on `scirs2-*` crates
4. **tensorlogic-train** depends on both `infer` (traits) and `scirs-backend` (concrete types)

---

## Compilation Pipeline

### Phase 1: Parsing (User Input → TLExpr)

```rust
// User writes high-level logic:
let rule = exists("x", "Person",
    and(
        pred("HasSkill", vec![var("x"), constant("Rust")]),
        pred("WantsJob", vec![var("x"), constant("Backend")])
    )
);
```

This produces a `TLExpr::Exists` AST node.

### Phase 2: Symbol Table Construction

The compiler needs to know:
- What domains exist (e.g., `Person`, `Skill`, `Job`)
- Variable bindings (`x: Person`)
- Predicate signatures (`HasSkill(Person, Skill)`)

```rust
let mut symbol_table = SymbolTable::new();
symbol_table.add_domain(DomainInfo::new("Person", 100))?;
symbol_table.add_domain(DomainInfo::new("Skill", 50))?;
symbol_table.infer_from_expr(&rule)?; // Infer predicates
```

### Phase 3: Static Analysis Passes

The compiler runs multiple passes over the TLExpr:

1. **Scope Analysis** (`passes/scope_analysis.rs`)
   - Verify all variables are bound
   - Detect shadowing and capture
   - Ensure quantifier ranges are valid

2. **Type Checking** (`passes/type_checking.rs`)
   - Check predicate arities match
   - Verify domain compatibility
   - Detect type mismatches

3. **Diagnostics** (`passes/diagnostics.rs`)
   - Detect unused variables
   - Find unreachable code
   - Report warnings

4. **Common Subexpression Elimination** (`passes/cse.rs`)
   - Identify repeated subexpressions
   - Create temporary nodes
   - Reduce redundant computation

5. **Strategy Selection** (`passes/strategy_selection.rs`)
   - Choose semantics (soft vs. hard logic)
   - Select quantifier reduction strategy (sum, max, product)
   - Assign tensor operation types

### Phase 4: Lowering to EinsumGraph

The compiler converts `TLExpr` to `EinsumGraph`:

```rust
// ∃x. P(x) ∧ Q(x)
// Becomes:
// temp0 = P(x)           # EinsumNode for P
// temp1 = Q(x)           # EinsumNode for Q
// temp2 = temp0 * temp1  # Hadamard product
// result = sum(temp2, axis=x)  # Reduction over domain
```

Each `EinsumNode` contains:
```rust
pub struct EinsumNode {
    pub id: NodeId,
    pub op: EinsumOp,               // Einsum, Hadamard, Sum, etc.
    pub inputs: Vec<NodeId>,        // Input node IDs
    pub output_shape: Vec<usize>,   // Shape inference
    pub einsum_spec: Option<String>,// Einsum notation (e.g., "ij,jk->ik")
    pub metadata: Option<NodeMetadata>, // Provenance, annotations
}
```

### Phase 5: Graph Optimization

The `GraphOptimizer` applies transformations:

1. **Algebraic Simplification** (`optimize/algebraic.rs`)
   - `NOT(NOT(x)) → x`
   - `AND(x, TRUE) → x`
   - `OR(x, FALSE) → x`

2. **Constant Folding** (`optimize/constant_folding.rs`)
   - Evaluate compile-time constants
   - Propagate known values

3. **Dead Code Elimination**
   - Remove unused nodes
   - Prune unreachable branches

4. **Operator Fusion** (`tensorlogic-infer/src/optimization.rs`)
   - Fuse elementwise operations
   - Reduce memory allocations
   - Minimize kernel launches (GPU)

### Phase 6: Placement and Scheduling

For multi-device execution:

```rust
let placement = PlacementOptimizer::new()
    .with_devices(vec![Device::CPU, Device::GPU(0)])
    .optimize(&graph)?;

let schedule = Scheduler::new()
    .with_placement(placement)
    .schedule(&graph)?;
```

This produces:
- Device assignments for each node
- Execution order respecting dependencies
- Data transfer points between devices

---

## Type System

### Domain Types

Domains represent sets of entities:

```rust
pub struct DomainInfo {
    pub name: String,
    pub cardinality: usize,        // Size of the domain
    pub description: Option<String>,
    pub elements: Option<Vec<String>>, // Explicit enumeration
}
```

**Design Choice:** We use cardinality (size) rather than explicit membership because:
1. Tensor dimensions need static sizes at compile-time
2. Large domains (millions of entities) can't be enumerated
3. Compatibility with neural networks (learned embeddings)

### Predicate Types

Predicates are multi-argument relations:

```rust
pub struct PredicateInfo {
    pub name: String,
    pub arity: usize,              // Number of arguments
    pub arg_domains: Vec<String>,  // Domain for each argument
    pub description: Option<String>,
}
```

Example:
```rust
// Predicate: Friends(Person, Person)
PredicateInfo {
    name: "Friends".to_string(),
    arity: 2,
    arg_domains: vec!["Person".to_string(), "Person".to_string()],
    description: Some("Symmetric friendship relation".to_string()),
}
```

This maps to a **2D tensor** of shape `[|Person|, |Person|]`.

### Term Types

```rust
pub enum Term {
    Variable(String),           // Quantified variable
    Constant(String),           // Ground constant
    DomainElement(String, usize), // Explicit domain element
}
```

**Design Choice:** We distinguish `Constant` (symbolic) from `DomainElement` (indexed) to support:
1. Symbolic manipulation before execution
2. Efficient tensor indexing during execution
3. Human-readable debugging

### Expression Types

The `TLExpr` enum is the core AST:

```rust
pub enum TLExpr {
    // Base predicates
    Predicate { name: String, args: Vec<Term> },

    // Logical connectives
    And { left: Box<TLExpr>, right: Box<TLExpr> },
    Or { left: Box<TLExpr>, right: Box<TLExpr> },
    Not { expr: Box<TLExpr> },
    Implies { left: Box<TLExpr>, right: Box<TLExpr> },

    // Quantifiers
    Exists { domain: String, var: String, body: Box<TLExpr> },
    ForAll { domain: String, var: String, body: Box<TLExpr> },

    // Arithmetic
    Add { left: Box<TLExpr>, right: Box<TLExpr> },
    Sub { left: Box<TLExpr>, right: Box<TLExpr> },
    Mul { left: Box<TLExpr>, right: Box<TLExpr> },
    Div { left: Box<TLExpr>, right: Box<TLExpr> },

    // Comparisons
    Eq { left: Box<TLExpr>, right: Box<TLExpr> },
    Lt { left: Box<TLExpr>, right: Box<TLExpr> },
    Gt { left: Box<TLExpr>, right: Box<TLExpr> },

    // Conditionals
    IfThenElse { cond: Box<TLExpr>, then_expr: Box<TLExpr>, else_expr: Box<TLExpr> },

    // Fuzzy logic (NEW in v0.1.0)
    TNorm { left: Box<TLExpr>, right: Box<TLExpr>, norm_type: String },
    TCoNorm { left: Box<TLExpr>, right: Box<TLExpr>, norm_type: String },
    FuzzyNot { expr: Box<TLExpr>, negation_type: String },

    // Soft quantifiers (NEW)
    SoftExists { domain: String, var: String, body: Box<TLExpr>, temperature: f64 },
    SoftForAll { domain: String, var: String, body: Box<TLExpr>, temperature: f64 },

    // Weighted rules (NEW)
    WeightedRule { expr: Box<TLExpr>, weight: f64 },

    // Probabilistic choice (NEW)
    ProbabilisticChoice { choices: Vec<(Box<TLExpr>, f64)> },

    // Temporal operators (NEW - LTL)
    Until { left: Box<TLExpr>, right: Box<TLExpr> },
    Release { left: Box<TLExpr>, right: Box<TLExpr> },
    WeakUntil { left: Box<TLExpr>, right: Box<TLExpr> },
    StrongRelease { left: Box<TLExpr>, right: Box<TLExpr> },
}
```

**Design Rationale:**
- Boxed recursive fields minimize enum size (1 pointer vs. multiple words)
- Explicit operator nodes enable pattern matching optimizations
- Separation of soft/hard variants allows strategy selection

---

## Execution Model

### Trait-Based Abstraction

The execution model is defined by traits in `tensorlogic-infer`:

```rust
pub trait TlExecutor: Send + Sync {
    type Tensor: Clone + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    fn execute(
        &self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
    ) -> Result<HashMap<String, Self::Tensor>, Self::Error>;

    fn execute_streaming(
        &self,
        graph: &EinsumGraph,
        inputs: impl Iterator<Item = HashMap<String, Self::Tensor>>,
        batch_size: usize,
    ) -> Result<Vec<HashMap<String, Self::Tensor>>, Self::Error>;
}

pub trait TlAutodiff: TlExecutor {
    fn backward(
        &self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
        grad_output: &Self::Tensor,
    ) -> Result<HashMap<String, Self::Tensor>, Self::Error>;
}
```

**Design Benefits:**
1. **Backend Portability**: Implement once, run on any backend
2. **Testing**: Mock implementations for unit tests
3. **Performance**: Monomorphization eliminates vtable overhead
4. **Extensibility**: New backends don't require core changes

### SciRS2 Backend Implementation

```rust
pub struct Scirs2Exec {
    cache: Arc<RwLock<TensorCache>>,
    config: ExecutionConfig,
}

impl TlExecutor for Scirs2Exec {
    type Tensor = Array<f64, IxDyn>;
    type Error = ExecutionError;

    fn execute(
        &self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
    ) -> Result<HashMap<String, Self::Tensor>, Self::Error> {
        // Topological execution order
        let schedule = self.schedule_graph(graph)?;

        let mut outputs = HashMap::new();

        for node_id in schedule {
            let node = &graph.nodes[node_id];

            // Fetch inputs from cache or outputs
            let input_tensors = self.get_input_tensors(node, &outputs)?;

            // Dispatch to operation
            let result = self.execute_node(node, &input_tensors)?;

            // Store result
            outputs.insert(node.id.clone(), result);

            // Update cache if enabled
            if self.config.caching_enabled {
                self.cache.write().unwrap().insert(node.id.clone(), outputs[&node.id].clone());
            }
        }

        Ok(outputs)
    }
}
```

**Key Implementation Details:**

1. **Lazy Evaluation**: Nodes are executed only when needed
2. **Caching**: Intermediate results cached to avoid recomputation
3. **Error Context**: Errors annotated with node ID for debugging

### Execution Strategies

Different compilation strategies affect execution:

| Strategy | Operator | Execution |
|----------|----------|-----------|
| **Hard Boolean** | AND → `min(a, b)` | Threshold at 0.5 |
| **Soft Differentiable** | AND → `a * b` | Gradient flows through |
| **Fuzzy Gödel** | AND → `min(a, b)` | No thresholding |
| **Fuzzy Product** | AND → `a * b` | Same as soft |
| **Fuzzy Łukasiewicz** | AND → `max(0, a + b - 1)` | Bounded arithmetic |
| **Probabilistic** | AND → Independent events | Probability calculus |

---

## Memory Management

### Tensor Lifecycle

```
┌─────────────┐
│ Allocation  │  (User inputs + intermediate nodes)
└──────┬──────┘
       │
┌──────▼──────┐
│  Execution  │  (Nodes evaluated in topological order)
└──────┬──────┘
       │
┌──────▼──────┐
│   Caching   │  (Optional: keep intermediates for reuse)
└──────┬──────┘
       │
┌──────▼──────┐
│ Deallocation│  (Drop tensors no longer needed)
└─────────────┘
```

### TensorCache

```rust
pub struct TensorCache {
    cache: HashMap<String, Array<f64, IxDyn>>,
    capacity: usize,
    policy: CachePolicy,
}

pub enum CachePolicy {
    LRU,          // Least recently used
    LFU,          // Least frequently used
    FIFO,         // First in, first out
    NoEviction,   // Never evict (unbounded)
}
```

**Design Choice:** We use `Arc<RwLock<TensorCache>>` for:
1. Thread-safe concurrent access
2. Shared ownership across executor instances
3. Interior mutability without `&mut self`

### Memory Pool

For high-performance workloads:

```rust
pub struct MemoryPool {
    allocator: Arc<dyn TensorAllocator>,
    pool_size: usize,
    allocated: AtomicUsize,
}

impl MemoryPool {
    pub fn allocate(&self, shape: &[usize]) -> Result<Array<f64, IxDyn>> {
        // Try to reuse from pool
        if let Some(tensor) = self.try_reuse(shape) {
            return Ok(tensor);
        }

        // Allocate new
        self.allocator.allocate(shape)
    }

    pub fn deallocate(&self, tensor: Array<f64, IxDyn>) {
        // Return to pool for reuse
        self.return_to_pool(tensor);
    }
}
```

**Benefits:**
- Reduces allocation overhead (critical for GPU)
- Predictable memory usage
- Avoids fragmentation

---

## Integration Patterns

### 1. OxiRS Bridge (RDF/SHACL → TensorLogic)

**Use Case:** Compile data governance rules into tensor constraints

```rust
// Parse SHACL shape
let schema = SchemaAnalyzer::from_turtle(turtle_data)?;

// Extract constraints
let shapes = schema.parse_shapes()?;

// Convert to TLExpr
let constraint_expr = shapes_to_tlexpr(&shapes)?;

// Compile to tensor graph
let graph = compile(&constraint_expr, &symbol_table)?;

// Execute validation
let violations = executor.execute(&graph, &data_tensors)?;
```

**Integration Point:** `tensorlogic-oxirs-bridge/src/shacl.rs`

### 2. QuantrS2 Hooks (PGM ↔ TensorLogic)

**Use Case:** Convert factor graphs to tensor operations

```rust
// Build factor graph from TLExpr
let factor_graph = FactorGraph::from_tlexpr(&expr)?;

// Run message passing
let beliefs = factor_graph.belief_propagation()?;

// Convert back to tensor format
let tensor_beliefs = beliefs_to_tensors(&beliefs)?;
```

**Integration Point:** `tensorlogic-quantrs-hooks/src/lib.rs`

### 3. SkleaRS Kernels (Logic-Derived Similarity)

**Use Case:** Use logical rules to define similarity metrics

```rust
// Define similarity rule
let similarity_rule = and(
    pred("SameCategory", vec![var("x"), var("y")]),
    pred("ClosePrices", vec![var("x"), var("y")])
);

// Compile to kernel
let kernel = LogicKernel::from_rule(&similarity_rule)?;

// Use in SkleaRS classifier
let svm = SVC::new().with_kernel(kernel);
```

**Integration Point:** `tensorlogic-sklears-kernels/src/lib.rs`

### 4. TrustformeRS (Transformers as Rules)

**Use Case:** Express transformer layers as logical rules

```rust
// Self-attention as einsum rule
let attention_rule = einsum("bhqd,bhkd->bhqk", &Q, &K);

// Compile with TensorLogic
let attention_graph = compile(&attention_rule)?;

// Execute with TrustformeRS runtime
let attention_weights = trustformers_executor.execute(&attention_graph)?;
```

**Integration Point:** `tensorlogic-trustformers/src/lib.rs`

---

## Performance Considerations

### 1. Compilation Time vs. Execution Time

**Trade-off:** Aggressive optimization increases compile time but reduces execution time.

**Strategy:**
- Development mode: Fast compilation, minimal optimization
- Production mode: Slow compilation, maximal optimization

```rust
let config = CompilationConfig {
    optimization_level: OptLevel::Aggressive, // O3 equivalent
    enable_cse: true,
    enable_fusion: true,
    enable_constant_folding: true,
};
```

### 2. Memory vs. Computation

**Trade-off:** Caching saves computation but increases memory usage.

**Strategy:**
- Small graphs: No caching (minimal overhead)
- Large graphs with reuse: LRU caching
- Streaming: No caching (unbounded inputs)

### 3. Parallelism Granularity

**Trade-off:** Fine-grained parallelism has overhead; coarse-grained may underutilize.

**Strategy:**
- Small tensors: Sequential execution
- Medium tensors: Thread-level parallelism (Rayon)
- Large tensors: SIMD + multi-threading
- Huge tensors: GPU offloading

### 4. Backend Selection

| Backend | Best For | Latency | Throughput |
|---------|----------|---------|------------|
| **CPU** | Small batches, low latency | 1-10ms | Low |
| **SIMD** | Medium batches, balanced | 0.5-5ms | Medium |
| **GPU** | Large batches, high throughput | 10-100ms | High |

**Heuristic:**
```rust
fn select_backend(batch_size: usize, tensor_size: usize) -> Backend {
    if batch_size < 32 && tensor_size < 10_000 {
        Backend::CPU
    } else if batch_size < 256 {
        Backend::SIMD
    } else {
        Backend::GPU
    }
}
```

---

## Extension Points

### 1. Adding New TLExpr Variants

**Steps:**
1. Add enum variant in `tensorlogic-ir/src/lib.rs`
2. Update all match statements in `tensorlogic-compiler/src/passes/*.rs`
3. Implement compilation logic in `tensorlogic-compiler/src/compile/mod.rs`
4. Add tests in `tensorlogic-compiler/tests/`

**Pattern Matching:** Use `cargo fix --allow-dirty` to find all match sites.

### 2. Implementing Custom Backends

**Steps:**
1. Create new crate: `tensorlogic-<backend>-backend`
2. Implement `TlExecutor` trait
3. Optionally implement `TlAutodiff` for training
4. Add feature flags for conditional compilation

**Example:**
```rust
pub struct OnnxBackend {
    session: onnxruntime::Session,
}

impl TlExecutor for OnnxBackend {
    type Tensor = onnxruntime::Tensor;
    type Error = OnnxError;

    fn execute(&self, graph: &EinsumGraph, inputs: &HashMap<String, Self::Tensor>)
        -> Result<HashMap<String, Self::Tensor>, Self::Error>
    {
        // Convert EinsumGraph to ONNX model
        let onnx_model = graph_to_onnx(graph)?;

        // Run ONNX inference
        self.session.run(onnx_model, inputs)
    }
}
```

### 3. Adding Custom Loss Functions

**Steps:**
1. Implement `Loss` trait in `tensorlogic-train/src/loss.rs`
2. Add gradient computation
3. Write unit tests

**Template:**
```rust
#[derive(Debug, Clone)]
pub struct CustomLoss {
    pub hyperparameter: f64,
}

impl Loss for CustomLoss {
    fn compute(&self, predictions: &ArrayView<f64, Ix2>, targets: &ArrayView<f64, Ix2>)
        -> TrainResult<f64>
    {
        // Implement forward pass
        todo!()
    }

    fn gradient(&self, predictions: &ArrayView<f64, Ix2>, targets: &ArrayView<f64, Ix2>)
        -> TrainResult<Array<f64, Ix2>>
    {
        // Implement backward pass
        todo!()
    }
}
```

### 4. Creating Custom Optimizers

**Steps:**
1. Implement `Optimizer` trait in `tensorlogic-train/src/optimizer.rs`
2. Manage optimizer state (momentum, adaptive rates)
3. Handle state serialization for checkpointing

**Template:**
```rust
pub struct CustomOptimizer {
    pub config: OptimizerConfig,
    pub state: HashMap<String, Array<f64, Ix2>>,
}

impl Optimizer for CustomOptimizer {
    fn step(&mut self, gradients: &HashMap<String, Array<f64, Ix2>>)
        -> TrainResult<HashMap<String, Array<f64, Ix2>>>
    {
        // Implement parameter update rule
        todo!()
    }

    fn zero_grad(&mut self) {
        // Clear accumulated gradients
        self.state.clear();
    }
}
```

---

## Design Decisions

### 1. Why Einsum as the Core Abstraction?

**Rationale:**
- **Expressiveness**: Einsum can represent most tensor operations
- **Optimization**: Well-studied fusion and scheduling algorithms
- **Backend Support**: NumPy, PyTorch, TensorFlow all have einsum
- **Readability**: Notation is concise and standard

**Alternative Considered:** Custom IR with explicit loops
- **Rejected Because:** Less portable, harder to optimize

### 2. Why Separate Planning and Execution Layers?

**Rationale:**
- **Portability**: Same logic can run on CPU, GPU, FPGA
- **Optimization**: Static analysis without runtime overhead
- **Serialization**: Compiled plans can be saved and loaded
- **Testing**: Mock executors for unit tests

**Alternative Considered:** Direct execution during compilation
- **Rejected Because:** Tight coupling, no backend abstraction

### 3. Why Trait-Based Executor Design?

**Rationale:**
- **Extensibility**: Users can implement custom backends
- **Monomorphization**: Zero-cost abstractions via generics
- **Testability**: Mock implementations for testing
- **Ecosystem Integration**: Each ecosystem crate can provide its own executor

**Alternative Considered:** Enum-based dispatch
- **Rejected Because:** Closed set of backends, runtime overhead

### 4. Why SymbolTable Instead of Type Inference?

**Rationale:**
- **Explicit Domains**: Users must declare domain sizes (tensor dimensions)
- **Error Messages**: Better diagnostics when types mismatch
- **Metadata**: Store descriptions, provenance, lineage
- **Interop**: JSON serialization for external tools

**Alternative Considered:** Hindley-Milner type inference
- **Rejected Because:** Can't infer tensor dimensions from logic alone

### 5. Why SciRS2 Integration Policy?

**Rationale:**
- **Consistency**: All COOLJAPAN projects use SciRS2
- **Performance**: SciRS2 is optimized for scientific computing
- **Features**: SIMD, GPU support, automatic differentiation
- **Maintenance**: Single source of truth for tensor operations

**Alternative Considered:** Direct ndarray usage
- **Rejected Because:** Fragmentation across ecosystem, missing features

### 6. Why Metadata on Every Node?

**Rationale:**
- **Debugging**: Trace back to source rules
- **Provenance**: Track data lineage for governance
- **Optimization**: Use hints for placement and scheduling
- **Reproducibility**: Store hyperparameters and versions

**Alternative Considered:** Separate metadata table
- **Rejected Because:** Easy to get out of sync, harder to propagate

### 7. Why Composable Loss/Regularization/Augmentation?

**Rationale:**
- **Flexibility**: Mix and match components
- **Reusability**: Share implementations across projects
- **Experimentation**: Quick prototyping of new combinations
- **Readability**: Clear intent in code

**Alternative Considered:** Monolithic training loops
- **Rejected Because:** Hard to extend, poor separation of concerns

---

## Future Directions

### 1. Distributed Execution

**Planned:** Automatic data/model parallelism across multiple machines

```rust
let executor = DistributedExecutor::new()
    .with_nodes(vec!["node1:8080", "node2:8080"])
    .with_strategy(ParallelismStrategy::DataParallel);
```

### 2. Incremental Compilation

**Planned:** Recompile only changed subgraphs

```rust
let diff = graph_diff(&old_graph, &new_graph)?;
let incremental_plan = recompile_diff(&diff)?;
```

### 3. Hardware-Specific Kernels

**Planned:** Custom kernels for TPU, FPGA, custom ASICs

```rust
#[target_arch = "tpu"]
impl TlExecutor for TpuBackend {
    // TPU-optimized einsum implementation
}
```

### 4. Probabilistic Programming Integration

**Planned:** Seamless interop with probabilistic languages (Stan, Pyro)

```rust
let prior = pyro::distributions::Normal::new(0.0, 1.0);
let likelihood = compile(&rule)?;
let posterior = pyro::infer(&likelihood, &prior, &data)?;
```

---

## Conclusion

TensorLogic's architecture is designed for:
- **Modularity**: Clear separation between planning and execution
- **Extensibility**: Trait-based abstractions for custom backends
- **Performance**: Zero-cost abstractions with SciRS2 integration
- **Ecosystem Integration**: First-class interop with COOLJAPAN projects

**Key Takeaways:**
1. Compilation is separate from execution
2. Traits define execution contracts
3. SciRS2 is the canonical numeric backend
4. Metadata propagates through all transformations
5. Composability is a first-class design principle

For implementation details, see:
- [GETTING_STARTED.md](GETTING_STARTED.md) - Practical usage guide
- [README.md](README.md) - Project overview
- [SCIRS2_INTEGRATION_POLICY.md](SCIRS2_INTEGRATION_POLICY.md) - Dependency rules

---

**Document Maintained By:** TensorLogic Contributors
**License:** MIT OR Apache-2.0
**Repository:** https://github.com/cool-japan/tensorlogic
