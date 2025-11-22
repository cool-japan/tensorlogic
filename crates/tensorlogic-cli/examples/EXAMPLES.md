# TensorLogic CLI Examples

This directory contains comprehensive examples demonstrating the TensorLogic CLI's capabilities, including compilation, optimization, and execution with multiple backends.

## Prerequisites

Make sure you have the TensorLogic CLI installed:

```bash
cargo install --path crates/tensorlogic-cli
# or
cargo build -p tensorlogic-cli --release
# Binary at: target/release/tensorlogic
```

## Examples Overview

### 01_backend_comparison.sh

**Purpose**: Compare execution performance across different backends.

**Features demonstrated**:
- Listing available backends
- Executing with CPU backend
- Executing with parallel backend (Rayon)
- Executing with profiled backend
- Exporting results in multiple formats (JSON, CSV, table)

**Usage**:
```bash
chmod +x examples/01_backend_comparison.sh
./examples/01_backend_comparison.sh
```

**Output**: Performance metrics for each backend, JSON and CSV result files.

### 02_optimization_pipeline.sh

**Purpose**: Demonstrate graph optimization at different levels.

**Features demonstrated**:
- Compilation without optimization
- Optimization levels: none, basic, standard, aggressive
- Optimization statistics (DCE, CSE, identity simplification)
- Verbose optimization output
- Graph visualization with Graphviz

**Usage**:
```bash
chmod +x examples/02_optimization_pipeline.sh
./examples/02_optimization_pipeline.sh
```

**Output**: Optimized graphs in JSON and DOT formats, PNG visualization (if Graphviz is installed).

### 03_repl_workflow.sh

**Purpose**: Demonstrate interactive REPL workflow.

**Features demonstrated**:
- Interactive REPL commands
- Domain management
- Backend selection
- Compilation, optimization, and execution in REPL
- Command history
- Context inspection

**Usage**:
```bash
chmod +x examples/03_repl_workflow.sh
./examples/03_repl_workflow.sh

# Or run REPL directly:
tensorlogic repl
```

**REPL Commands**:
```
tensorlogic> .help              # Show all commands
tensorlogic> .domain Person 100 # Add domain
tensorlogic> knows(x, y)        # Compile expression
tensorlogic> .execute --metrics # Execute last compiled graph
tensorlogic> .optimize aggressive --stats  # Optimize
tensorlogic> .backend parallel  # Change backend
tensorlogic> .exit              # Exit REPL
```

### 04_complete_pipeline.sh

**Purpose**: End-to-end workflow from expression to optimized execution.

**Features demonstrated**:
- Complex social network reasoning example
- Multi-step pipeline: compile → analyze → optimize → visualize → execute
- Performance comparison across backends
- Result export in multiple formats
- Automated workflow orchestration

**Usage**:
```bash
chmod +x examples/04_complete_pipeline.sh
./examples/04_complete_pipeline.sh
```

**Output**: Complete set of analysis artifacts (graphs, visualizations, performance data).

## Example Workflows

### Quick Start: Basic Compilation

```bash
# Simple predicate
tensorlogic "person(x)"

# Logical operations
tensorlogic "knows(x, y) AND likes(y, z)"

# With domains
tensorlogic "EXISTS x IN Person. knows(x, alice)" \
  --domain Person:100

# With analysis
tensorlogic "complex_expression(x)" \
  --analyze \
  --output-format stats
```

### Optimization Workflow

```bash
# Basic optimization
tensorlogic optimize "p(x) AND q(y) AND p(x)" \
  --level basic \
  --stats

# Aggressive optimization with visualization
tensorlogic optimize "complex_expr(x)" \
  --level aggressive \
  --stats \
  --verbose \
  --output-format dot | dot -Tpng -o optimized.png
```

### Execution Workflow

```bash
# Execute with default backend (CPU)
tensorlogic execute "knows(x, y)" \
  --domain Person:100 \
  --metrics

# Execute with parallel backend
tensorlogic execute "knows(x, y)" \
  --domain Person:100 \
  --backend parallel \
  --metrics \
  --output-format json

# List available backends
tensorlogic backends
```

### REPL Workflow

```bash
$ tensorlogic repl

tensorlogic> .domain Person 100
✓ Added domain 'Person' with size 100

tensorlogic> .strategy soft_differentiable
✓ Strategy set to: soft_differentiable

tensorlogic> knows(x, y) AND lives_in(x, city)
✓ Compilation successful
  3 tensors, 3 nodes, depth 2

tensorlogic> .optimize aggressive --stats
ℹ Optimizing with aggressive level...
✓ Optimization complete

tensorlogic> .execute --metrics
ℹ Executing with SciRS2 CPU backend...
✓ Execution completed in 1.234 ms

tensorlogic> .backend parallel
✓ Backend set to: SciRS2 Parallel

tensorlogic> .execute --metrics
ℹ Executing with SciRS2 Parallel backend...
✓ Execution completed in 0.856 ms
```

## Advanced Usage

### Combining with Other Tools

#### With jq (JSON processing)

```bash
# Extract execution time
tensorlogic execute "expr" --output-format json | jq '.execution_time_ms'

# Compare backends
for backend in cpu parallel; do
    tensorlogic execute "expr" --backend $backend --output-format json \
        | jq -r ".backend, .execution_time_ms"
done
```

#### With Graphviz (Visualization)

```bash
# Generate and view visualization
tensorlogic "complex_expr" --output-format dot \
    | dot -Tsvg | display  # or open in browser
```

#### With Python (NumPy format)

```bash
# Export for NumPy
tensorlogic execute "expr" --output-format numpy > results.txt

# Load in Python:
# import numpy as np
# data = np.loadtxt('results.txt', comments='#')
```

## Performance Tips

1. **Use parallel backend for large graphs**
   ```bash
   tensorlogic execute "large_expr" --backend parallel --metrics
   ```

2. **Optimize before execution**
   ```bash
   # Compile and save optimized graph
   tensorlogic optimize "expr" --level aggressive -o optimized.json

   # Execute optimized graph
   tensorlogic execute optimized.json -f json --backend parallel
   ```

3. **Profile execution**
   ```bash
   tensorlogic execute "expr" --backend profiled --metrics
   ```

4. **Batch processing**
   ```bash
   # Create expressions file
   cat > expressions.txt << EOF
   knows(x, y)
   likes(y, z)
   knows(x, y) AND likes(y, z)
   EOF

   # Process batch
   tensorlogic batch expressions.txt
   ```

## Troubleshooting

### Example scripts don't execute

Make sure scripts are executable:
```bash
chmod +x examples/*.sh
```

### Graphviz visualization not working

Install Graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Other systems
# See: https://graphviz.org/download/
```

### Backend not available

Check available backends:
```bash
tensorlogic backends
```

If SIMD or GPU backends are not shown, they need to be enabled at compile time:
```bash
# Build with SIMD support
cargo build -p tensorlogic-cli --features simd

# Build with GPU support (future)
cargo build -p tensorlogic-cli --features gpu
```

## Next Steps

- Read the [CLI README](../README.md) for complete command reference
- Try the interactive REPL: `tensorlogic repl`
- Explore different compilation strategies
- Experiment with your own logical expressions
- Profile and optimize your workloads

## Contributing Examples

Have a useful workflow or example? Contributions are welcome!

1. Create a new example script in this directory
2. Add documentation in this README
3. Ensure the example is self-contained and well-commented
4. Submit a pull request

---

**Part of the TensorLogic Project**
For more information, see the [main documentation](../../README.md).
