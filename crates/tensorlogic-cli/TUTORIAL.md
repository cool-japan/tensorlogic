# TensorLogic CLI Tutorial

**Complete guide to using TensorLogic CLI for logic-to-tensor compilation**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Workflows](#workflows)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Reference](#reference)

---

## Introduction

### What is TensorLogic CLI?

TensorLogic CLI is a command-line tool and library for compiling logical expressions (predicates, quantifiers, logical operators) into executable tensor computation graphs (einsum operations). It bridges symbolic logic and numerical tensor computation, enabling:

- **Neural-symbolic AI**: Train neural networks with logical constraints
- **Differentiable reasoning**: Backpropagate through logical rules
- **Knowledge graph queries**: Execute graph queries as tensor operations
- **Constraint satisfaction**: Express constraints as differentiable functions

### Why Use TensorLogic?

**Traditional Approach:**
```
Logic Rules → Symbolic Solver → Boolean Results
```
- Not differentiable
- Hard to integrate with ML pipelines
- Discrete outputs only

**TensorLogic Approach:**
```
Logic Rules → Tensor Graph → Continuous/Discrete Outputs
```
- Fully differentiable
- Native tensor operations (GPU-ready)
- Supports fuzzy logic, probabilities, and Boolean logic
- Integrates seamlessly with PyTorch, TensorFlow, JAX

---

## Installation

### From crates.io (Recommended)

```bash
cargo install tensorlogic-cli
```

### From Source

```bash
git clone https://github.com/cool-japan/tensorlogic.git
cd tensorlogic
cargo install --path crates/tensorlogic-cli
```

### Verify Installation

```bash
tensorlogic --version
tensorlogic --help
```

---

## Quick Start

### Your First Compilation

```bash
# Compile a simple predicate
tensorlogic "knows(alice, bob)"
```

**Output:**
```
✓ Compilation successful
  Graph: 1 tensors, 0 nodes, 0 inputs, 1 outputs

EinsumGraph {
    tensors: ["knows[ab]"],
    nodes: [],
    inputs: [],
    outputs: [0],
}
```

### Logical Operations

```bash
# AND operation
tensorlogic "knows(x, y) AND likes(y, z)"

# OR operation
tensorlogic "person(x) OR robot(x)"

# NOT operation
tensorlogic "NOT mortal(x)"

# Implication
tensorlogic "knows(x, y) -> likes(x, y)"
```

### With Analysis

```bash
tensorlogic "knows(x, y) AND likes(y, z)" --analyze
```

**Output:**
```
Graph Analysis Metrics:
  Tensors: 3
  Nodes: 1
  Depth: 1
  Estimated FLOPs: 20,000
  Estimated Memory: 80,000 bytes
```

---

## Core Concepts

### 1. Logical Expressions

TensorLogic supports standard first-order logic:

```
Predicates:    pred(x, y)
Operators:     AND, OR, NOT, IMPLIES
Quantifiers:   EXISTS x IN Domain. expr
               FORALL x IN Domain. expr
Arithmetic:    x + y, x * y, x / y
Comparisons:   x = y, x < y, x > y, x <= y, x >= y
Conditionals:  IF cond THEN x ELSE y
```

### 2. Compilation Strategies

Six strategies map logic to tensor operations:

| Strategy | Use Case | AND | OR | NOT |
|----------|----------|-----|-----|-----|
| **soft_differentiable** | Neural training | `a * b` | `a + b - ab` | `1 - a` |
| **hard_boolean** | Discrete logic | `min(a,b)` | `max(a,b)` | `1 - a` |
| **fuzzy_godel** | Gödel fuzzy logic | `min(a,b)` | `max(a,b)` | `1 - a` |
| **fuzzy_product** | Product fuzzy logic | `a * b` | `a + b - ab` | `1 - a` |
| **fuzzy_lukasiewicz** | Łukasiewicz logic | `max(0,a+b-1)` | `min(1,a+b)` | `1 - a` |
| **probabilistic** | Probabilities | `a * b` | `a + b - ab` | `1 - a` |

### 3. Domains

Domains define the universe of discourse for variables:

```bash
--domains Person:100    # Person domain with 100 individuals
--domains City:50       # City domain with 50 cities
```

Default domain `D` is created automatically if not specified.

### 4. Einsum Graphs

Output format showing tensor computations:

```
EinsumGraph {
    tensors: [list of tensors],
    nodes: [computation nodes],
    inputs: [input indices],
    outputs: [output indices]
}
```

---

## Basic Usage

### Example 1: Social Network

```bash
tensorlogic "EXISTS x. (knows(alice, x) AND knows(x, bob))" \
  --domains Person:100 \
  --strategy soft_differentiable \
  --validate \
  --analyze
```

**Meaning**: Does Alice know someone who knows Bob? (Friend of friend)

### Example 2: Access Control

```bash
tensorlogic "admin(user) OR (owns(user, resource) AND NOT locked(resource))" \
  --domains User:1000 \
  --domains Resource:5000 \
  --strategy hard_boolean \
  --output-format json \
  --output access_policy.json
```

**Meaning**: Access granted if user is admin OR (owns resource AND resource not locked)

### Example 3: Recommendation

```bash
tensorlogic "likes(user, item) OR EXISTS x. (likes(user, x) AND similar(x, item))" \
  --domains User:10000 \
  --domains Item:50000 \
  --strategy probabilistic \
  --output-format stats
```

**Meaning**: Recommend items user likes OR similar to items they like

---

## Advanced Features

### Interactive REPL

```bash
tensorlogic repl
```

```
TensorLogic Interactive REPL
Type '.help' for commands, '.exit' to quit

tensorlogic> .domain Person 100
✓ Added domain 'Person' with size 100

tensorlogic> .strategy fuzzy_godel
✓ Strategy set to: fuzzy_godel

tensorlogic> EXISTS x IN Person. knows(x, alice)
✓ Compilation successful
  Graph: 2 tensors, 2 nodes, depth 2

tensorlogic> .history
   1: .domain Person 100
   2: .strategy fuzzy_godel
   3: EXISTS x IN Person. knows(x, alice)

tensorlogic> .exit
```

### Batch Processing

Create `rules.txt`:
```
knows(alice, bob)
likes(bob, charlie)
friend(alice, charlie)
# Transitive friendship
EXISTS x. (friend(alice, x) AND friend(x, charlie))
```

Process all rules:
```bash
tensorlogic batch rules.txt
```

### Watch Mode

Auto-recompile on file changes:

```bash
tensorlogic watch my_rules.tl
```

### Visualization

Generate Graphviz diagrams:

```bash
tensorlogic "knows(x, y) AND likes(y, z)" \
  --output-format dot > graph.dot

dot -Tpng graph.dot -o graph.png
dot -Tsvg graph.dot -o graph.svg
```

### Optimization

Optimize compiled graphs:

```bash
tensorlogic optimize "complex_expression(x, y, z)" \
  --input-format expr \
  --level aggressive \
  --stats \
  --verbose
```

Optimization levels:
- **none**: No optimization
- **basic**: Dead code elimination
- **standard**: DCE + identity simplification
- **aggressive**: All passes + reordering

### Benchmarking

Measure compilation performance:

```bash
tensorlogic benchmark "knows(x, y) AND likes(y, z)" \
  --iterations 100 \
  --execute \
  --backend scirs2-cpu \
  --json > benchmark_results.json
```

### Profiling

Detailed performance breakdown:

```bash
tensorlogic profile "complex_rule(x, y)" \
  --warmup 5 \
  --runs 20 \
  --validate \
  --json
```

---

## Workflows

### Workflow 1: Development Cycle

```bash
# 1. Create rule file
cat > policy.tl << 'EOF'
FORALL u IN User. (
  admin(u) OR (
    verified(u) AND NOT suspended(u)
  )
)
EOF

# 2. Watch for changes
tensorlogic watch policy.tl &

# 3. Edit policy.tl in your editor
# 4. See live compilation results
# 5. When satisfied, validate
tensorlogic policy.tl --validate --analyze

# 6. Export for use
tensorlogic policy.tl --output-format json --output policy.json
```

### Workflow 2: Performance Tuning

```bash
# 1. Baseline benchmark
tensorlogic benchmark my_rule.tl --iterations 100 > baseline.txt

# 2. Compare strategies
for strategy in soft_differentiable hard_boolean fuzzy_godel; do
  echo "=== $strategy ==="
  tensorlogic my_rule.tl --strategy $strategy --output-format stats
done

# 3. Profile bottlenecks
tensorlogic profile my_rule.tl --verbose

# 4. Optimize
tensorlogic optimize my_rule.tl --level aggressive --stats
```

### Workflow 3: Integration Testing

```bash
# test_rules.sh
#!/bin/bash

RULES_DIR="rules/"
TESTS_PASSED=0
TESTS_FAILED=0

for rule_file in "$RULES_DIR"/*.tl; do
  echo "Testing $rule_file..."

  if tensorlogic "$rule_file" --validate --quiet; then
    echo "✓ PASS: $rule_file"
    ((TESTS_PASSED++))
  else
    echo "✗ FAIL: $rule_file"
    ((TESTS_FAILED++))
  fi
done

echo ""
echo "Results: $TESTS_PASSED passed, $TESTS_FAILED failed"
[ $TESTS_FAILED -eq 0 ]
```

### Workflow 4: Documentation Generation

```bash
# Generate visual documentation for all rules
mkdir -p docs/graphs

for rule in rules/*.tl; do
  basename="${rule%.tl}"

  # Generate DOT
  tensorlogic "$rule" --output-format dot > "docs/graphs/${basename}.dot"

  # Generate PNG
  dot -Tpng "docs/graphs/${basename}.dot" -o "docs/graphs/${basename}.png"

  # Generate stats
  tensorlogic "$rule" --output-format stats > "docs/graphs/${basename}_stats.txt"
done
```

---

## Best Practices

### 1. Start Simple

```bash
# ✓ Good: Start with simple expressions
tensorlogic "knows(x, y)"
tensorlogic "knows(x, y) AND likes(y, z)"

# ✗ Avoid: Complex expressions from the start
tensorlogic "FORALL x. EXISTS y. ((a(x,y) AND b(y,z)) OR (c(x) AND d(z)))"
```

### 2. Use Appropriate Strategies

```bash
# Neural training: soft_differentiable
tensorlogic rule.tl --strategy soft_differentiable

# Boolean logic: hard_boolean
tensorlogic rule.tl --strategy hard_boolean

# Probabilities: probabilistic
tensorlogic rule.tl --strategy probabilistic
```

### 3. Always Validate During Development

```bash
tensorlogic rule.tl --validate --analyze
```

### 4. Use Domains Explicitly

```bash
# ✓ Good: Explicit domains
tensorlogic "EXISTS x IN Person. knows(x, alice)" \
  --domains Person:100

# ✗ Avoid: Relying on default domain
tensorlogic "EXISTS x. knows(x, alice)"
```

### 5. Profile Before Optimizing

```bash
# 1. Profile to find bottlenecks
tensorlogic profile rule.tl --verbose

# 2. Then optimize
tensorlogic optimize rule.tl --level aggressive
```

### 6. Use Quiet Mode in Scripts

```bash
#!/bin/bash
if tensorlogic rule.tl --validate --quiet; then
  echo "Validation passed"
else
  echo "Validation failed"
  exit 1
fi
```

---

## Troubleshooting

### Issue: "Compilation failed"

**Cause**: Syntax error in expression

**Solution**:
```bash
# Use debug mode to see details
tensorlogic "your expression" --debug
```

### Issue: "Validation failed: Free variable"

**Cause**: Unbound variable in expression

**Solution**:
```bash
# Add quantifier or domain
tensorlogic "EXISTS x IN Domain. your_expr" --domains Domain:100
```

### Issue: "Unknown compilation strategy"

**Cause**: Typo in strategy name

**Solution**:
```bash
# List valid strategies
tensorlogic --help | grep strategy

# Use correct name
tensorlogic rule.tl --strategy soft_differentiable
```

### Issue: Performance is slow

**Solutions**:
```bash
# 1. Reduce domain sizes during development
tensorlogic rule.tl --domains Person:10  # Instead of 1000

# 2. Use optimization
tensorlogic optimize rule.tl --level aggressive

# 3. Profile to find bottlenecks
tensorlogic profile rule.tl --verbose

# 4. Try different strategies
tensorlogic rule.tl --strategy hard_boolean  # Often faster than soft
```

### Issue: Out of memory

**Solutions**:
```bash
# 1. Reduce domain sizes
--domains User:100  # Instead of 10000

# 2. Simplify expression
# Break complex rules into smaller pieces

# 3. Use streaming/batch processing
tensorlogic batch rules.txt
```

---

## Reference

### Command Quick Reference

```bash
# Basic compilation
tensorlogic "expression" [OPTIONS]

# Subcommands
tensorlogic repl                      # Interactive mode
tensorlogic batch FILES...            # Batch processing
tensorlogic watch FILE                # Watch mode
tensorlogic completion SHELL          # Shell completion
tensorlogic config show|path|init     # Configuration
tensorlogic convert FILE              # Format conversion
tensorlogic execute EXPR              # Execute with backend
tensorlogic optimize EXPR             # Optimize graph
tensorlogic backends                  # List backends
tensorlogic benchmark EXPR            # Benchmark
tensorlogic profile EXPR              # Profile
tensorlogic cache stats|clear         # Cache management
```

### Common Options

```
-f, --input-format FORMAT    # expr, json, yaml
-o, --output FILE           # Output file
-F, --output-format FORMAT  # graph, dot, json, stats
-s, --strategy STRATEGY     # Compilation strategy
-d, --domains NAME:SIZE     # Domain definition
--validate                  # Enable validation
--debug                     # Debug output
-a, --analyze              # Show metrics
-q, --quiet                # Quiet mode
--no-color                 # Disable colors
```

### Configuration File

Location: `~/.tensorlogicrc` or `./.tensorlogicrc`

```toml
# Default compilation strategy
strategy = "soft_differentiable"

# Enable colored output
colored = true

# Enable validation by default
validate = false

# Default domains
[domains]
Person = 100
City = 50

# REPL settings
[repl]
prompt = "tensorlogic> "
history_file = ".tensorlogic_history"
max_history = 1000

# Watch mode settings
[watch]
debounce_ms = 500
clear_screen = true
show_timestamps = true
```

Initialize configuration:
```bash
tensorlogic config init
tensorlogic config edit
```

---

## Next Steps

1. **Try the examples**: Explore `examples/` directory
2. **Read the cookbook**: See [COOKBOOK.md](COOKBOOK.md) for recipes
3. **API documentation**: `cargo doc --open -p tensorlogic-cli`
4. **Join the community**: [GitHub Discussions](https://github.com/cool-japan/tensorlogic/discussions)

---

**Happy Compiling!** 🚀
