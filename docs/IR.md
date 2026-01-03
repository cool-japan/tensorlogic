# Tensorlogic Intermediate Representation (IR)

## Overview

The Tensorlogic IR is a **planning-layer representation** that bridges logical expressions (`TLExpr`) and executable tensor computation graphs (`EinsumGraph`). The IR is designed to be:

- **Engine-agnostic**: No dependency on specific tensor backends
- **Serializable**: Full serde support for caching and debugging
- **Analyzable**: Static analysis for optimization and validation
- **Composable**: Graphs can be merged and transformed

## Core Types

### Term

The most basic unit representing variables or constants:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Term {
    Var(String),    // Bound variable
    Const(String),  // Constant value
}
```

**Invariants**:
- Variable names must be valid identifiers
- Constants are treated as singleton domains

### TLExpr

The abstract syntax tree for logical expressions:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TLExpr {
    Pred { name: String, args: Vec<Term> },
    And(Box<TLExpr>, Box<TLExpr>),
    Or(Box<TLExpr>, Box<TLExpr>),
    Not(Box<TLExpr>),
    Exists { var: String, domain: String, body: Box<TLExpr> },
    ForAll { var: String, domain: String, body: Box<TLExpr> },
    Imply(Box<TLExpr>, Box<TLExpr>),
    Score(Box<TLExpr>),
}
```

**Key Properties**:
- Predicates are n-ary relations over terms
- Quantifiers bind variables to specific domains
- `Score` wraps soft-logic expressions for optimization
- All variants are recursively composable

### EinsumNode

A single einsum operation in the computation graph:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EinsumNode {
    pub spec: String,       // Einstein notation (e.g., "ij,jk->ik")
    pub inputs: Vec<usize>, // Indices to input tensors
}
```

**Spec Format**:
- Input subscripts separated by commas
- Arrow `->` followed by output subscripts
- Implicit sum over repeated indices
- Examples:
  - `"ij,jk->ik"` - Matrix multiplication
  - `"ij,ij->ij"` - Hadamard product
  - `"ijk->ij"` - Sum reduction over k

**Invariants**:
- `spec` must be valid einsum notation
- `inputs.len()` must match number of input tensors in spec
- All input indices must reference valid tensors

### EinsumGraph

The complete computation graph:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EinsumGraph {
    pub tensors: Vec<String>,    // Tensor names/IDs
    pub nodes: Vec<EinsumNode>,  // Computation nodes
    pub outputs: Vec<usize>,     // Indices of output tensors
}
```

**Graph Structure**:
- **Tensors**: Named tensor slots (inputs + intermediates)
- **Nodes**: Ordered sequence of operations
- **Outputs**: Final result tensor indices

**Execution Semantics**:
- Nodes execute in declaration order
- Each node may depend on previous nodes
- Outputs are subset of all tensors

## Compilation Pipeline

### Phase 1: Expression Analysis

Input: `TLExpr`

Analysis steps:
1. **Free Variable Collection**: Identify all unbound variables
2. **Domain Inference**: Determine domain for each variable
3. **Arity Checking**: Validate predicate argument counts
4. **Type Checking**: Ensure domain consistency

Output: Validated AST + symbol table

### Phase 2: Axis Assignment

Input: Validated `TLExpr` + domain bindings

Process:
1. Assign each unique variable to a tensor axis
2. Build axis metadata (domain name, cardinality)
3. Compute tensor shapes for predicates
4. Track axis correspondence across operations

Output: Axis assignment map + tensor shapes

### Phase 3: Graph Emission

Input: `TLExpr` + axis assignments

Process for each expression type:

**Predicate**:
```rust
Pred { name: "P", args: [Var("x"), Var("y")] }
  → Tensor: "P[x,y]"
  → Shape: [|domain(x)|, |domain(y)|]
  → No einsum node (input tensor)
```

**AND**:
```rust
And(A, B)  // A: shape [i,j], B: shape [i,j]
  → Node: { spec: "ij,ij->ij", inputs: [A_idx, B_idx] }
  → Output: Hadamard product
```

**OR** (max-based):
```rust
Or(A, B)
  → Intermediate1: max(A, B)  // element-wise max
  → Or soft variant: A + B - A*B
```

**NOT**:
```rust
Not(A)
  → Node: { spec: "ij->ij", inputs: [A_idx] }
  → Apply element-wise: 1 - A
```

**EXISTS** (sum reduction):
```rust
Exists { var: "z", body: P(x,z,y) }  // P: shape [i,k,j]
  → Node: { spec: "ikj->ij", inputs: [P_idx] }
  → Sum over k axis
```

**FORALL** (dual of exists):
```rust
ForAll { var: "z", body: P(x,z) }
  → Desugar to: NOT(EXISTS { var: "z", body: NOT(P(x,z)) })
  → Apply NOT and EXISTS rules
```

**IMPLY**:
```rust
Imply(A, B)
  → ReLU(B - A)  // or max(0, B - A)
  → Or soft: max(1 - A, B)
```

Output: Complete `EinsumGraph`

## Example: Parent to Ancestor

### Input Expression

```rust
// Parent(x, y) → Ancestor(x, y)
let rule = TLExpr::Imply(
    Box::new(TLExpr::Pred {
        name: "Parent".into(),
        args: vec![Term::Var("x".into()), Term::Var("y".into())],
    }),
    Box::new(TLExpr::Pred {
        name: "Ancestor".into(),
        args: vec![Term::Var("x".into()), Term::Var("y".into())],
    }),
);
```

### Compiled IR

```rust
EinsumGraph {
    tensors: vec![
        "Parent[x,y]".into(),      // Index 0: input
        "Ancestor[x,y]".into(),    // Index 1: input/output
        "temp_relu".into(),        // Index 2: intermediate
    ],
    nodes: vec![
        // Compute: Ancestor - Parent
        EinsumNode {
            spec: "ij,ij->ij".into(),
            inputs: vec![1, 0],  // Subtract operation
        },
        // Apply ReLU
        EinsumNode {
            spec: "ij->ij".into(),
            inputs: vec![2],  // ReLU(temp)
        },
    ],
    outputs: vec![2],  // Final ReLU result
}
```

## Static Analysis

### Validation Checks

1. **Arity Consistency**: All uses of same predicate have same arity
2. **Domain Consistency**: Same variable always bound to same domain
3. **Axis Alignment**: Binary operations have compatible shapes
4. **Graph Connectivity**: All nodes reference valid tensor indices
5. **Output Validity**: Output indices exist in tensor list

### Optimization Opportunities

1. **Common Subexpression Elimination**: Reuse identical subexpressions
2. **Constant Folding**: Evaluate constant expressions at compile time
3. **Redundant Computation**: Eliminate NOT(NOT(A)) → A
4. **Axis Permutation**: Optimize einsum axis ordering
5. **Fusion**: Combine consecutive element-wise operations

## Metadata Structures

### Symbol Table

```rust
pub struct SymbolTable {
    pub predicates: HashMap<String, PredicateInfo>,
    pub domains: HashMap<String, DomainInfo>,
    pub variables: HashMap<String, VariableInfo>,
}

pub struct PredicateInfo {
    pub arity: usize,
    pub arg_domains: Vec<String>,
}

pub struct DomainInfo {
    pub cardinality: usize,
    pub elements: Option<Vec<String>>,
}

pub struct VariableInfo {
    pub domain: String,
    pub axis: Option<usize>,
}
```

### Axis Metadata

```rust
pub struct AxisMetadata {
    pub var_to_axis: HashMap<String, usize>,
    pub axis_to_domain: HashMap<usize, String>,
    pub shapes: HashMap<String, Vec<usize>>,
}
```

## Serialization Format

The IR uses serde for JSON/bincode serialization:

```json
{
  "tensors": ["Parent[x,y]", "Ancestor[x,y]"],
  "nodes": [
    {
      "spec": "ij,ij->ij",
      "inputs": [0, 1]
    }
  ],
  "outputs": [1]
}
```

This enables:
- **Caching**: Save compiled graphs for reuse
- **Debugging**: Inspect intermediate representations
- **Distributed Execution**: Serialize for remote execution
- **Interoperability**: Exchange with other tools

## Design Rationale

### Why Einsum?

1. **Universality**: Einsum can express any tensor contraction
2. **Clarity**: Notation makes data flow explicit
3. **Optimization**: Backends can optimize einsum paths
4. **Differentiability**: Automatic differentiation support

### Engine Agnosticism

The IR deliberately avoids:
- Concrete tensor types (ndarray, Tensor, etc.)
- Execution semantics (CPU, GPU, distributed)
- Numerical precision (f32, f64, etc.)

This allows:
- Multiple backend implementations
- Testing with lightweight mocks
- Backend-specific optimizations

## Future Extensions

Planned IR enhancements:

1. **Control Flow**: If-then-else, switch statements
2. **Loops**: Fixed-point iteration for recursive rules
3. **Probabilistic**: Stochastic operations, sampling
4. **Sparse Tensors**: Compact representation for sparse data
5. **Type System**: Rich type annotations for domains
6. **Provenance**: Track rule application lineage

## References

- Einstein summation: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
- Datalog compilation techniques
- Tensor network representations

---

**Status**: Draft v0.1
****Last Updated**: 2025-12-16
