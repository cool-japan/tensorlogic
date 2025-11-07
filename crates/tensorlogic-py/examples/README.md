# TensorLogic Python Examples

Comprehensive examples demonstrating all features of pytensorlogic.

## Quick Navigation

| Example | Description | Difficulty | Features |
|---------|-------------|------------|----------|
| **basic_usage.py** | Complete usage guide | ⭐ Beginner | All core operations |
| **arithmetic_operations.py** | Arithmetic operations | ⭐ Beginner | Add, sub, mul, div |
| **comparison_conditionals.py** | Comparisons & conditionals | ⭐⭐ Intermediate | gt, lt, if-then-else |
| **advanced_symbol_table.py** | Domain management | ⭐⭐ Intermediate | SymbolTable, domains |
| **backend_selection.py** | Backend capabilities | ⭐⭐ Intermediate | CPU, SIMD selection |
| **provenance_tracking.py** | Full provenance workflow | ⭐⭐⭐ Advanced | RDF*, SHACL, tracking |

## Running Examples

```bash
# Basic example
python examples/basic_usage.py

# Specific feature
python examples/provenance_tracking.py

# Run all examples
for script in examples/*.py; do python "$script"; done
```

## Examples by Feature

### Core Features

#### basic_usage.py (350+ lines)
Complete introduction to pytensorlogic covering:
- Creating logical expressions (predicates, variables, constants)
- Logical operations (AND, OR, NOT, implication)
- Quantifiers (EXISTS, FORALL)
- Compilation and execution
- NumPy integration

**What you'll learn**:
- How to build logical rules
- How to compile expressions
- How to execute with NumPy data
- Graph statistics and analysis

**Run it**:
```bash
python examples/basic_usage.py
```

### Arithmetic & Comparisons

#### arithmetic_operations.py (200+ lines)
All arithmetic operations with examples:
- Addition (`add`)
- Subtraction (`sub`)
- Multiplication (`mul`)
- Division (`div`)
- Combined operations

**What you'll learn**:
- Performing arithmetic on predicates
- Chaining operations
- Working with constants
- Execution with numeric data

**Run it**:
```bash
python examples/arithmetic_operations.py
```

#### comparison_conditionals.py (280+ lines)
Comparison operators and conditional logic:
- Equality (`eq`)
- Less than (`lt`)
- Greater than (`gt`)
- Less/greater or equal (`lte`, `gte`)
- Conditional expressions (`if_then_else`)
- Nested conditionals

**What you'll learn**:
- Comparing values in logical rules
- Building conditional logic
- Nested conditionals
- Real-world classification tasks

**Run it**:
```bash
python examples/comparison_conditionals.py
```

### Domain Management

#### advanced_symbol_table.py (400+ lines)
Advanced schema management with symbol tables:
- Creating domains with metadata
- Defining predicate signatures
- Variable bindings
- Automatic schema inference
- JSON serialization
- Real-world social network example

**What you'll learn**:
- Building rich semantic models
- Domain metadata management
- Schema import/export
- Multi-stage compilation

**Run it**:
```bash
python examples/advanced_symbol_table.py
```

### Backend Selection

#### backend_selection.py (280+ lines)
Backend capabilities and selection:
- Listing available backends
- Querying backend capabilities
- Backend-specific execution
- Performance comparison
- System information

**What you'll learn**:
- How to choose the right backend
- Performance optimization
- Feature detection
- SIMD acceleration

**Run it**:
```bash
python examples/backend_selection.py
```

### Provenance Tracking

#### provenance_tracking.py (450+ lines)
Complete provenance tracking workflow with RDF* integration:
- Source location tracking
- Provenance metadata
- RDF entity mappings
- SHACL shape tracking
- RDF* triple tracking with confidence
- High-confidence inference filtering
- Turtle export
- JSON persistence
- Real-world social network reasoning

**What you'll learn**:
- Tracking computation origins
- RDF/SHACL integration
- Confidence-based filtering
- Audit trail creation
- Provenance export formats

**Run it**:
```bash
python examples/provenance_tracking.py
```

## Learning Path

### 1. Start Here (Beginners)
1. `basic_usage.py` - Learn the fundamentals
2. `arithmetic_operations.py` - Understand arithmetic
3. `comparison_conditionals.py` - Master conditionals

### 2. Intermediate Level
4. `advanced_symbol_table.py` - Domain management
5. `backend_selection.py` - Performance tuning

### 3. Advanced Topics
6. `provenance_tracking.py` - Full provenance workflow

## Common Patterns

### Creating Predicates
```python
import pytensorlogic as tl

# Binary predicate
knows = tl.pred("knows", [tl.var("x"), tl.var("y")])

# Unary predicate
person = tl.pred("Person", [tl.var("x")])

# With constants
alice_knows_bob = tl.pred("knows", [
    tl.const("alice"),
    tl.const("bob")
])
```

### Combining Rules
```python
# AND
rule = tl.and_(person_x, knows)

# OR
rule = tl.or_(knows, knows_reverse)

# Chaining multiple rules
rules = [rule1, rule2, rule3]
combined = rules[0]
for r in rules[1:]:
    combined = tl.and_(combined, r)
```

### Quantifiers
```python
# EXISTS: "there exists a y such that..."
exists_rule = tl.exists("y", "Person", knows)

# FORALL: "for all y..."
forall_rule = tl.forall("y", "Person", knows)

# Nested quantifiers
nested = tl.forall("x", "Person",
    tl.exists("y", "Person", knows)
)
```

### Execution Pattern
```python
# 1. Build expression
expr = tl.pred("knows", [x, y])

# 2. Compile
graph = tl.compile(expr)

# 3. Prepare inputs
inputs = {
    "knows": np.array([[...]])  # Your data here
}

# 4. Execute
result = tl.execute(graph, inputs)

# 5. Use output
output = result["output"]
```

## Tips & Tricks

### Performance
```python
# Use SIMD for better performance
result = tl.execute(graph, inputs, backend=tl.Backend.SCIRS2_SIMD)

# Check what's available
backends = tl.list_available_backends()
print(backends)
```

### Debugging
```python
# Check graph structure
stats = graph.stats()
print(f"Nodes: {stats['num_nodes']}")
print(f"Outputs: {stats['num_outputs']}")

# Extract metadata
metadata = tl.get_metadata(graph)
for i, meta in enumerate(metadata):
    if meta:
        print(f"Node {i}: {meta}")
```

### Reusability
```python
# Save symbol table
table = tl.symbol_table()
# ... configure table ...
json_data = table.to_json()

# Later, restore
restored_table = tl.SymbolTable.from_json(json_data)
```

## Requirements

All examples require:
```bash
pip install numpy
pip install maturin

# Build pytensorlogic
cd crates/pytensorlogic
maturin develop
```

## Troubleshooting

**Import Error**: Make sure pytensorlogic is installed
```bash
maturin develop
python -c "import pytensorlogic; print('OK')"
```

**Runtime Error**: Check your input shapes match expected dimensions
```python
print(graph.stats())
print({k: v.shape for k, v in inputs.items()})
```

**Build Error**: Ensure Rust toolchain is installed
```bash
rustc --version
cargo --version
```

## Next Steps

After exploring the examples:

1. **Read the full API reference** in `README.md`
2. **Check out the tutorials** in `tutorials/` (Jupyter notebooks)
3. **Run the test suite** to see more usage patterns:
   ```bash
   pytest tests/ -v
   ```
4. **Build your own application** using pytensorlogic!

## Contributing

Found a bug or have an improvement?
- Open an issue: https://github.com/cool-japan/tensorlogic/issues
- Submit a pull request with your example!

---

**Happy coding! 🚀**
