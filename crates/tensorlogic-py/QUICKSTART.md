# TensorLogic Python - Quick Start Guide

Get up and running with TensorLogic in under 5 minutes!

## Installation

```bash
# Install maturin
pip install maturin

# Build and install
cd crates/pytensorlogic
maturin develop
```

## Your First Rule

```python
import pytensorlogic as tl
import numpy as np

# Define: "x knows y"
x, y = tl.var("x"), tl.var("y")
knows = tl.pred("knows", [x, y])

# Compile
graph = tl.compile(knows)

# Execute with data
knows_matrix = np.random.rand(100, 100)
result = tl.execute(graph, {"knows": knows_matrix})
print(result["output"].shape)  # (100, 100)
```

## Common Patterns

### Quantifiers

```python
# EXISTS: "x knows someone"
knows_someone = tl.exists("y", "Person", knows)

# FORALL: "everyone knows x"
everyone_knows_x = tl.forall("y", "Person", knows)
```

### Logic Operations

```python
# AND: "x is a person AND x knows y"
person_x = tl.pred("Person", [x])
person_knows_someone = tl.and_(person_x, knows)

# OR: "x knows y OR y knows x"
knows_mutual = tl.or_(knows, tl.pred("knows", [y, x]))

# NOT: "x doesn't know y"
doesnt_know = tl.not_(knows)

# IMPLICATION: "if x knows y then y knows x"
symmetric = tl.imply(knows, tl.pred("knows", [y, x]))
```

### Arithmetic & Comparisons

```python
# Arithmetic
age = tl.pred("age", [x])
older = tl.add(age, tl.constant(5.0))

# Comparisons
adult = tl.gt(age, tl.constant(18.0))

# Conditionals
status = tl.if_then_else(
    adult,
    tl.constant(1.0),  # adult
    tl.constant(0.0)   # minor
)
```

## Compilation Strategies

Choose the right semantics for your use case:

```python
# Neural network training (default)
config = tl.CompilationConfig.soft_differentiable()

# Discrete Boolean logic
config = tl.CompilationConfig.hard_boolean()

# Fuzzy logic
config = tl.CompilationConfig.fuzzy_godel()

# Use config
graph = tl.compile_with_config(expr, config)
```

## Backend Selection

Get better performance with the right backend:

```python
# Check what's available
backends = tl.list_available_backends()
print(backends)

# Use SIMD for 2-4x speedup
result = tl.execute(graph, inputs, backend=tl.Backend.SCIRS2_SIMD)

# Get backend info
caps = tl.get_backend_capabilities(tl.Backend.SCIRS2_CPU)
print(f"{caps.name} v{caps.version}")
print(f"Features: {caps.features}")
```

## Domain Management

Build semantic models:

```python
# Create symbol table
table = tl.symbol_table()

# Define domain
person_domain = tl.domain_info("Person", cardinality=100)
person_domain.set_description("All people in the network")
table.add_domain(person_domain)

# Define predicate
knows_pred = tl.predicate_info("knows", ["Person", "Person"])
table.add_predicate(knows_pred)

# Bind variables
table.bind_variable("x", "Person")

# Save/load
json_data = table.to_json()
restored = tl.SymbolTable.from_json(json_data)
```

## Provenance Tracking

Track the origin of inferences:

```python
# Create tracker
tracker = tl.provenance_tracker(enable_rdfstar=True)

# Track entities
tracker.track_entity("http://example.org/alice", 0)

# Track inferred triples with confidence
tracker.track_inferred_triple(
    subject="http://example.org/alice",
    predicate="http://example.org/knows",
    object="http://example.org/bob",
    rule_id="rule_1",
    confidence=0.95
)

# Get high-confidence inferences
high_conf = tracker.get_high_confidence_inferences(min_confidence=0.85)

# Export to RDF* Turtle
turtle = tracker.to_rdfstar_turtle()
```

## Next Steps

1. **Explore examples**: Check out `examples/` directory
   ```bash
   python examples/basic_usage.py
   python examples/provenance_tracking.py
   ```

2. **Read the docs**: See `README.md` for complete API reference

3. **Run tests**: Verify your installation
   ```bash
   pytest tests/ -v
   ```

4. **Check out tutorials**: See `tutorials/` for Jupyter notebooks

## Common Issues

**Build failed?**
```bash
# Make sure you have Rust installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Try clean build
rm -rf target/
maturin develop --release
```

**Import error?**
```bash
# Make sure you're in the right environment
which python
pip list | grep tensorlogic
```

**Shape mismatch?**
```python
# Check graph stats
stats = graph.stats()
print(f"Expected: {stats}")

# Check your input shapes
for name, arr in inputs.items():
    print(f"{name}: {arr.shape}")
```

## Get Help

- 📖 **Full Documentation**: See `README.md`
- 💡 **Examples**: Browse `examples/` directory
- 🧪 **Tests**: Check `tests/` for usage patterns
- 🐛 **Issues**: https://github.com/cool-japan/tensorlogic/issues

---

**Happy Logic Programming! 🎉**
