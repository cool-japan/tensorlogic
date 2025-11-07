# Tensorlogic Provenance Tracking

## Overview

Provenance in Tensorlogic refers to the **complete lineage** of tensor computations, from source logical rules through intermediate compilation steps to final tensor outputs. This enables:

- **Explainability**: Trace why a specific output value was computed
- **Debugging**: Identify which rules contribute to unexpected results
- **Auditing**: Verify computation integrity for compliance
- **Trust**: Build confidence in neural-symbolic hybrid models

Provenance tracking is **critical** for the OxiRS bridge and COOLJAPAN ecosystem integration.

## Provenance Levels

### Level 0: Rule Provenance

Track which logical rules produced which parts of the graph.

**Metadata**:
```rust
pub struct RuleProvenance {
    pub rule_id: String,              // Unique rule identifier
    pub rule_source: String,          // Original rule text/location
    pub timestamp: i64,               // When rule was compiled
    pub author: Option<String>,       // Who wrote the rule
}
```

**Example**:
```rust
Rule ID: "rule_001"
Source: "Parent(x,y) → Ancestor(x,y)"
Timestamp: 2025-11-03T10:00:00Z
Author: "system"
```

### Level 1: Graph Node Provenance

Map each `EinsumNode` to the `TLExpr` that generated it.

**Metadata**:
```rust
pub struct NodeProvenance {
    pub node_id: usize,               // Index in EinsumGraph.nodes
    pub expr_type: String,            // "AND", "EXISTS", etc.
    pub source_rule_id: String,       // Parent rule
    pub parent_nodes: Vec<usize>,     // Dependency nodes
}
```

**Example**:
```rust
Node ID: 2
Expression: "AND"
Source Rule: "rule_001"
Parents: [0, 1]  // Inputs from nodes 0 and 1
```

### Level 2: Tensor Provenance

Track origin of each tensor in the computation.

**Metadata**:
```rust
pub struct TensorProvenance {
    pub tensor_id: usize,             // Index in EinsumGraph.tensors
    pub name: String,                 // Tensor name
    pub predicate: Option<String>,    // Source predicate if input
    pub producer_node: Option<usize>, // Node that created this tensor
    pub data_source: DataSource,      // Where data came from
}

pub enum DataSource {
    InputData { source: String },     // External data file/DB
    Derived { from_tensors: Vec<usize> }, // Computed from others
    Learned { optimizer: String },    // Learned parameters
}
```

**Example**:
```rust
Tensor ID: 0
Name: "Parent[x,y]"
Predicate: Some("Parent")
Producer: None  // Input tensor
Data Source: InputData { source: "rdf://kb/parent_relations" }
```

### Level 3: Value Provenance

Trace individual tensor elements back to source data.

**Metadata** (sparse, on-demand):
```rust
pub struct ValueProvenance {
    pub tensor_id: usize,
    pub indices: Vec<usize>,          // Multi-dimensional index
    pub value: f64,
    pub contributors: Vec<Contribution>,
}

pub struct Contribution {
    pub source_tensor: usize,
    pub source_indices: Vec<usize>,
    pub operation: String,            // "multiply", "sum", etc.
    pub weight: f64,                  // Contribution strength
}
```

**Example**:
```rust
Tensor: 3 (output)
Index: [5, 10]
Value: 0.87
Contributors:
  - Source: Tensor 0[5,10], Op: "multiply", Weight: 0.45
  - Source: Tensor 1[5,10], Op: "multiply", Weight: 0.42
```

## OxiRS Bridge Integration

The `tensorlogic-oxirs-bridge` crate implements provenance binding to RDF* and SPARQL.

### RDF* Representation

Provenance stored as reified RDF triples:

```turtle
@prefix tl: <http://tensorlogic.cool-japan.org/vocab#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

# Rule provenance
<rule:001> a tl:LogicRule ;
    tl:ruleText "Parent(x,y) → Ancestor(x,y)" ;
    prov:generatedAtTime "2025-11-03T10:00:00Z"^^xsd:dateTime ;
    prov:wasAttributedTo <user:system> .

# Node provenance
<node:002> a tl:EinsumNode ;
    tl:nodeIndex 2 ;
    tl:operationType "AND" ;
    tl:sourceRule <rule:001> ;
    tl:dependsOn <node:000>, <node:001> .

# Tensor provenance
<tensor:000> a tl:Tensor ;
    tl:tensorName "Parent[x,y]" ;
    tl:predicate "Parent" ;
    prov:hadPrimarySource <data:kb/parent_relations> .

# Value provenance (RDF*)
<<tensor:003[5,10] tl:hasValue 0.87>> {
    prov:wasDerivedFrom <tensor:000[5,10]> ;
    tl:contributionWeight 0.45 ;
    tl:operation "multiply" .
}
```

### SPARQL Queries

Query provenance via SPARQL:

```sparql
# Find all rules that contributed to a specific output
PREFIX tl: <http://tensorlogic.cool-japan.org/vocab#>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?rule ?ruleText ?contribution
WHERE {
  <tensor:output[5,10]> prov:wasDerivedFrom+ ?intermediate .
  ?node tl:producedTensor ?intermediate ;
        tl:sourceRule ?rule .
  ?rule tl:ruleText ?ruleText .
  ?intermediate tl:contributionWeight ?contribution .
}
ORDER BY DESC(?contribution)
```

### SHACL Constraints

Validate provenance integrity:

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix tl: <http://tensorlogic.cool-japan.org/vocab#> .

# Every node must reference a valid rule
tl:NodeProvenanceShape a sh:NodeShape ;
    sh:targetClass tl:EinsumNode ;
    sh:property [
        sh:path tl:sourceRule ;
        sh:minCount 1 ;
        sh:class tl:LogicRule ;
    ] .

# Every tensor must have a data source
tl:TensorProvenanceShape a sh:NodeShape ;
    sh:targetClass tl:Tensor ;
    sh:or (
        [ sh:property [ sh:path tl:predicate ; sh:minCount 1 ] ]
        [ sh:property [ sh:path tl:producedBy ; sh:minCount 1 ] ]
    ) .
```

## Provenance Storage

### In-Memory (Development)

During compilation, store provenance in parallel structures:

```rust
pub struct ProvenanceStore {
    pub rule_prov: HashMap<String, RuleProvenance>,
    pub node_prov: HashMap<usize, NodeProvenance>,
    pub tensor_prov: HashMap<usize, TensorProvenance>,
    pub value_prov: Option<SparseValueStore>,  // Expensive, optional
}
```

### Persistent (Production)

For production deployments:

1. **RDF Store**: OxiGraph backend via OxiRS bridge
2. **Graph DB**: Neo4j, TigerGraph for traversal queries
3. **Time-Series DB**: InfluxDB for training lineage
4. **Append Log**: Kafka for immutable audit trail

### Serialization

Provenance serializes alongside IR:

```json
{
  "graph": { /* EinsumGraph */ },
  "provenance": {
    "rules": {
      "rule_001": {
        "source": "Parent(x,y) → Ancestor(x,y)",
        "timestamp": 1730635200,
        "author": "system"
      }
    },
    "nodes": {
      "0": {
        "expr_type": "Pred",
        "source_rule_id": "rule_001",
        "parents": []
      },
      "2": {
        "expr_type": "AND",
        "source_rule_id": "rule_001",
        "parents": [0, 1]
      }
    },
    "tensors": {
      "0": {
        "name": "Parent[x,y]",
        "predicate": "Parent",
        "data_source": {
          "InputData": { "source": "rdf://kb/parent_relations" }
        }
      }
    }
  }
}
```

## Provenance API

### Compilation-Time

```rust
use tensorlogic_compiler::{compile_to_einsum, CompilerOptions};

let options = CompilerOptions {
    track_provenance: true,
    provenance_level: ProvenanceLevel::Node,
    ..Default::default()
};

let result = compile_to_einsum_with_options(&expr, &options)?;

// Access provenance
for (node_id, prov) in result.provenance.nodes {
    println!("Node {}: from {}", node_id, prov.expr_type);
}
```

### Runtime

```rust
use tensorlogic_scirs_backend::Scirs2Exec;

let mut executor = Scirs2Exec::new_with_provenance();
let output = executor.run(&graph)?;

// Query value provenance
let prov = executor.trace_value(&output, &[5, 10])?;
for contrib in prov.contributors {
    println!("  From tensor {}: weight {}", contrib.source_tensor, contrib.weight);
}
```

### OxiRS Query

```rust
use tensorlogic_oxirs_bridge::ProvenanceQuery;

let query = ProvenanceQuery::new(oxigraph_store);

// Find all rules contributing to output
let rules = query.trace_rules(tensor_id, indices)?;
for rule in rules {
    println!("Rule: {} (contribution: {})", rule.text, rule.weight);
}

// Validate provenance integrity
let violations = query.validate_shacl(&shacl_graph)?;
assert!(violations.is_empty());
```

## Training Provenance

For learned models, track optimization lineage:

```rust
pub struct TrainingProvenance {
    pub run_id: String,
    pub hyperparameters: HashMap<String, f64>,
    pub optimizer: String,
    pub loss_function: String,
    pub iterations: Vec<IterationRecord>,
}

pub struct IterationRecord {
    pub iteration: usize,
    pub loss: f64,
    pub gradients: HashMap<usize, f64>,  // Tensor ID → gradient norm
    pub parameter_updates: Vec<ParameterUpdate>,
}

pub struct ParameterUpdate {
    pub tensor_id: usize,
    pub indices: Vec<usize>,
    pub old_value: f64,
    pub new_value: f64,
    pub gradient: f64,
}
```

This enables:
- Reproducing training runs
- Understanding parameter evolution
- Debugging convergence issues
- Compliance with model cards

## Privacy and Security

### Redaction

Sensitive data can be redacted from provenance:

```rust
let options = CompilerOptions {
    track_provenance: true,
    redact_values: true,  // Only track structure, not actual values
    redact_sources: vec!["sensitive_db"],  // Anonymize specific sources
    ..Default::default()
};
```

### Access Control

Provenance queries respect OxiRS access control:

```rust
// Only authorized users can trace to sensitive sources
let query = ProvenanceQuery::new_with_auth(store, user_credentials);
let trace = query.trace_value(tensor_id, indices)?;  // May return redacted results
```

### Differential Privacy

For public provenance, add noise:

```rust
let options = TrainingOptions {
    dp_epsilon: 1.0,  // Privacy budget
    dp_clip_norm: 1.0,
    track_provenance: true,
};
```

## Performance Considerations

Provenance tracking overhead:

| Level | Memory Overhead | Runtime Overhead |
|-------|----------------|------------------|
| Rule | ~1 KB/rule | Negligible |
| Node | ~100 B/node | <1% |
| Tensor | ~500 B/tensor | <5% |
| Value | ~10 MB/1000 values | 10-20% |

**Recommendations**:
- Use Level 1 (Node) by default
- Enable Value provenance only for debugging
- Use sampling for large-scale training
- Store provenance asynchronously

## Future Work

Planned enhancements:

1. **Incremental Tracking**: Update provenance without full recompilation
2. **Compression**: Deduplicate common provenance patterns
3. **Visualization**: Interactive provenance graphs
4. **Federated Provenance**: Track across distributed computations
5. **Counterfactual Queries**: "What if rule X was different?"

## References

- W3C PROV: https://www.w3.org/TR/prov-overview/
- RDF*: https://w3c.github.io/rdf-star/
- SHACL: https://www.w3.org/TR/shacl/
- Provenance in ML: https://arxiv.org/abs/2011.11184

---

**Status**: Draft v0.1
**Last Updated**: 2025-11-03
