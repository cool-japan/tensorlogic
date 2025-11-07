# Tensorlogic DSL Specification

## Overview

Tensorlogic provides a domain-specific language (DSL) for expressing logical rules that compile to tensor operations. The DSL is designed to be minimal, expressive, and directly mappable to Einstein summation notation.

## Syntax Elements

### Terms

Terms represent variables or constants in logical expressions:

```rust
pub enum Term {
    Var(String),      // Logical variable (e.g., "x", "person")
    Const(String),    // Constant value (e.g., "alice", "bob")
}
```

**Examples**:
- `Var("x")` - A variable ranging over some domain
- `Const("alice")` - A specific constant

### Logical Expressions

The core expression type supports first-order logic constructs:

```rust
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

#### Predicates

Predicates represent relationships between terms:

```rust
TLExpr::Pred {
    name: "Parent".into(),
    args: vec![Term::Var("x".into()), Term::Var("y".into())],
}
```

Represents: `Parent(x, y)`

#### Logical Connectives

**Conjunction (AND)**:
```rust
TLExpr::And(
    Box::new(expr1),
    Box::new(expr2),
)
```
Represents: `expr1 ∧ expr2`

**Disjunction (OR)**:
```rust
TLExpr::Or(
    Box::new(expr1),
    Box::new(expr2),
)
```
Represents: `expr1 ∨ expr2`

**Negation (NOT)**:
```rust
TLExpr::Not(Box::new(expr))
```
Represents: `¬expr`

**Implication**:
```rust
TLExpr::Imply(
    Box::new(premise),
    Box::new(conclusion),
)
```
Represents: `premise → conclusion`

#### Quantifiers

**Existential Quantification**:
```rust
TLExpr::Exists {
    var: "x".into(),
    domain: "Person".into(),
    body: Box::new(predicate),
}
```
Represents: `∃x ∈ Person. predicate(x)`

**Universal Quantification**:
```rust
TLExpr::ForAll {
    var: "x".into(),
    domain: "Person".into(),
    body: Box::new(predicate),
}
```
Represents: `∀x ∈ Person. predicate(x)`

## Semantic Interpretation

### Tensor Mapping

Each logical construct maps to a tensor operation:

| Logic | Tensor Operation | Notes |
|-------|-----------------|-------|
| `P(x)` | Tensor indexed by domain(x) | Shape: `[|domain|]` |
| `P(x,y)` | 2D tensor | Shape: `[|domain(x)|, |domain(y)|]` |
| `A ∧ B` | Hadamard product `A * B` | Element-wise multiplication |
| `A ∨ B` | `max(A, B)` | Or soft max: `A + B - A*B` |
| `¬A` | `1 - A` | Complement operation |
| `∃x. P(x)` | `sum(P, axis=x)` | Or `max` for hard quantification |
| `∀x. P(x)` | `NOT(∃x. NOT(P(x)))` | Dual of existential |
| `A → B` | `ReLU(B - A)` | Or soft: `max(1-A, B)` |

### Domain Binding

Variables must be bound to domains before tensor compilation:

```rust
// Domain cardinalities
let domains = HashMap::from([
    ("Person", 100),
    ("City", 50),
]);

// Variable bindings
let bindings = HashMap::from([
    ("x", "Person"),
    ("y", "City"),
]);
```

### Axis Assignment

Variables map to tensor axes based on their binding:

```
Pred("LivesIn", [Var("x"), Var("y")])
  → Tensor shape: [100, 50]
  → Axes: {x: 0, y: 1}
```

## Usage Examples

### Example 1: Transitive Closure

Expressing ancestor relationship from parent relationship:

```rust
// Parent(x, y) → Ancestor(x, y)
let base_case = TLExpr::Imply(
    Box::new(TLExpr::Pred {
        name: "Parent".into(),
        args: vec![Term::Var("x".into()), Term::Var("y".into())],
    }),
    Box::new(TLExpr::Pred {
        name: "Ancestor".into(),
        args: vec![Term::Var("x".into()), Term::Var("y".into())],
    }),
);

// Parent(x, z) ∧ Ancestor(z, y) → Ancestor(x, y)
let recursive_case = TLExpr::Imply(
    Box::new(TLExpr::And(
        Box::new(TLExpr::Pred {
            name: "Parent".into(),
            args: vec![Term::Var("x".into()), Term::Var("z".into())],
        }),
        Box::new(TLExpr::Pred {
            name: "Ancestor".into(),
            args: vec![Term::Var("z".into()), Term::Var("y".into())],
        }),
    )),
    Box::new(TLExpr::Pred {
        name: "Ancestor".into(),
        args: vec![Term::Var("x".into()), Term::Var("y".into())],
    }),
);
```

### Example 2: Existential Query

Find people who have at least one friend in a specific city:

```rust
// ∃z. Friend(x, z) ∧ LivesIn(z, "Tokyo")
let query = TLExpr::Exists {
    var: "z".into(),
    domain: "Person".into(),
    body: Box::new(TLExpr::And(
        Box::new(TLExpr::Pred {
            name: "Friend".into(),
            args: vec![Term::Var("x".into()), Term::Var("z".into())],
        }),
        Box::new(TLExpr::Pred {
            name: "LivesIn".into(),
            args: vec![Term::Var("z".into()), Term::Const("Tokyo".into())],
        }),
    )),
};
```

### Example 3: Universal Constraint

All employees must have a manager:

```rust
// ∀x ∈ Employee. ∃y ∈ Manager. ReportsTo(x, y)
let constraint = TLExpr::ForAll {
    var: "x".into(),
    domain: "Employee".into(),
    body: Box::new(TLExpr::Exists {
        var: "y".into(),
        domain: "Manager".into(),
        body: Box::new(TLExpr::Pred {
            name: "ReportsTo".into(),
            args: vec![Term::Var("x".into()), Term::Var("y".into())],
        }),
    }),
};
```

## Compilation Process

The DSL expressions compile to tensor graphs through these steps:

1. **Parsing/Construction**: Build `TLExpr` AST
2. **Type Checking**: Validate domain bindings, arity
3. **Axis Assignment**: Map variables to tensor dimensions
4. **Graph Emission**: Generate `EinsumGraph` with einsum specifications
5. **Execution**: Run on backend (SciRS2, etc.)

## Design Principles

1. **Minimalism**: Small core language, extensible through predicates
2. **Composability**: Expressions nest naturally
3. **Type Safety**: Domain checking at compile time
4. **Backend Agnostic**: DSL independent of execution engine
5. **Differentiability**: All operations support backpropagation

## Future Extensions

Planned DSL extensions:

- **Aggregations**: `count`, `sum`, `avg` over domains
- **Arithmetic**: Numeric operations in predicates
- **Temporal Logic**: `Next`, `Until`, `Always`, `Eventually`
- **Probabilistic**: Soft logic operators with learned weights
- **Meta-predicates**: Higher-order predicates

## References

- Tensor Logic paper: https://arxiv.org/abs/2510.12269
- Einstein summation notation
- First-order logic textbooks

---

**Status**: Draft v0.1
**Last Updated**: 2025-11-03
