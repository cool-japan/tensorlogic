# TensorLogic CLI Cookbook

**Practical recipes and best practices for using the TensorLogic CLI**

## Table of Contents

- [Basic Recipes](#basic-recipes)
- [Compilation Strategies](#compilation-strategies)
- [Format Conversion](#format-conversion)
- [Visualization](#visualization)
- [Batch Processing](#batch-processing)
- [Development Workflows](#development-workflows)
- [Integration Patterns](#integration-patterns)
- [Optimization Tips](#optimization-tips)
- [Troubleshooting](#troubleshooting)

## Basic Recipes

### Recipe 1: Compile a Simple Rule

```bash
tensorlogic "knows(alice, bob)"
```

**Output**: Compiled tensor graph representation

**Use case**: Quick compilation and validation

---

### Recipe 2: Compile with Validation

```bash
tensorlogic "knows(x, y) AND likes(y, z)" --validate --analyze
```

**Output**: Validated graph with complexity metrics

**Use case**: Ensure correctness before deployment

---

### Recipe 3: Save Compiled Graph

```bash
tensorlogic "person(x) -> mortal(x)" \
  --output-format json \
  --output compiled_rule.json
```

**Use case**: Save compiled graphs for later use

---

## Compilation Strategies

### Recipe 4: Neural Network Training

```bash
tensorlogic "student(x) AND smart(x) -> scholar(x)" \
  --strategy soft_differentiable \
  --output-format json
```

**Why**: Smooth gradients for backpropagation

**Use case**: Integrating with neural architectures

---

### Recipe 5: Discrete Boolean Logic

```bash
tensorlogic "authorized(u) AND NOT blocked(u) -> access(u)" \
  --strategy hard_boolean \
  --validate
```

**Why**: Crisp true/false decisions

**Use case**: Access control, binary classification

---

### Recipe 6: Fuzzy Reasoning

```bash
tensorlogic "temperature(x) > 30 AND humidity(x) > 70" \
  --strategy fuzzy_godel
```

**Why**: Handle uncertainty and partial truth

**Use case**: Environmental monitoring, expert systems

---

### Recipe 7: Probabilistic Inference

```bash
tensorlogic "FORALL x IN User. active(x) -> recommended(x, item)" \
  --strategy probabilistic \
  --domains User:10000
```

**Why**: Probabilistic interpretations

**Use case**: Recommendation systems, Bayesian networks

---

## Format Conversion

### Recipe 8: Expression to JSON

```bash
tensorlogic convert "knows(x, y) AND likes(y, z)" \
  --from expr \
  --to json \
  --pretty
```

**Use case**: Serialize for storage or transmission

---

### Recipe 9: JSON to Pretty Expression

```bash
tensorlogic convert rule.json \
  --from json \
  --to expr \
  --pretty
```

**Use case**: Human-readable documentation

---

### Recipe 10: Format Pipeline

```bash
# Convert expr -> JSON -> YAML -> expr
tensorlogic convert "p AND q" --from expr --to json --output step1.json
tensorlogic convert step1.json --from json --to yaml --output step2.yaml
tensorlogic convert step2.yaml --from yaml --to expr --pretty
```

**Use case**: Format transformation pipelines

---

## Visualization

### Recipe 11: Generate Graph Visualization

```bash
tensorlogic "FORALL x. (edge(x,y) AND edge(y,z)) -> path(x,z)" \
  --domains Node:50 \
  --output-format dot \
  --output graph.dot

dot -Tpng graph.dot -o graph.png
dot -Tsvg graph.dot -o graph.svg
```

**Use case**: Visual debugging, documentation

---

### Recipe 12: Interactive Visualization

```bash
tensorlogic "complex(rule)" --output-format dot | \
  dot -Tx11
```

**Use case**: Real-time graph exploration

---

## Batch Processing

### Recipe 13: Process Multiple Rules

Create `rules.txt`:
```
# Social network rules
knows(x, y) AND knows(y, z) -> might_know(x, z)
friend(x, y) -> friend(y, x)
FORALL x. person(x) -> mortal(x)

# Access control
admin(u) -> can_access(u, r)
owner(u, r) AND NOT locked(r) -> can_access(u, r)
```

Process:
```bash
tensorlogic batch rules.txt
```

**Use case**: Compile rule sets, batch validation

---

### Recipe 14: Batch with Custom Output

```bash
tensorlogic batch rules.txt \
  --output-format stats > batch_report.txt
```

**Use case**: Generate compilation reports

---

## Development Workflows

### Recipe 15: Watch Mode for Iterative Development

```bash
tensorlogic watch my_rules.tl
```

Edit `my_rules.tl` and save - automatic recompilation!

**Use case**: Rapid prototyping

---

### Recipe 16: REPL for Exploration

```bash
tensorlogic repl
```

```
tensorlogic> .domain Person 100
✓ Added domain 'Person' with size 100

tensorlogic> .strategy fuzzy_godel
✓ Strategy set to: fuzzy_godel

tensorlogic> knows(x, y) AND likes(y, z)
✓ Compilation successful
  3 tensors, 3 nodes, depth 2

tensorlogic> .history
   1: .domain Person 100
   2: .strategy fuzzy_godel
   3: knows(x, y) AND likes(y, z)
```

**Use case**: Interactive experimentation

---

### Recipe 17: Configuration Management

```bash
# Initialize config
tensorlogic config init

# Edit config
tensorlogic config edit

# Show current config
tensorlogic config show
```

**Use case**: Persistent settings across sessions

---

## Integration Patterns

### Recipe 18: Integration with jq

```bash
tensorlogic "knows(x, y)" --output-format json | \
  jq '.tensors | length'
```

**Use case**: Extract specific information from compiled graphs

---

### Recipe 19: Integration with Make

Create `Makefile`:
```makefile
.PHONY: compile validate visualize clean

RULES := $(wildcard rules/*.tl)
GRAPHS := $(RULES:.tl=.json)

compile: $(GRAPHS)

%.json: %.tl
	tensorlogic $< --output-format json --output $@ --validate

validate:
	tensorlogic batch $(RULES)

visualize: graph.png

graph.png: rules/main.tl
	tensorlogic $< --output-format dot | dot -Tpng -o $@

clean:
	rm -f $(GRAPHS) graph.png
```

Usage:
```bash
make compile
make validate
make visualize
```

**Use case**: Build automation

---

### Recipe 20: CI/CD Integration (GitHub Actions)

Create `.github/workflows/validate-rules.yml`:
```yaml
name: Validate TensorLogic Rules

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install TensorLogic
        run: cargo install tensorlogic-cli

      - name: Validate all rules
        run: tensorlogic batch rules/*.tl --validate

      - name: Generate statistics
        run: |
          for rule in rules/*.tl; do
            tensorlogic "$rule" --output-format stats --analyze
          done
```

**Use case**: Automated validation in CI pipeline

---

## Optimization Tips

### Recipe 21: Analyze Complexity

```bash
tensorlogic "complex(nested(rule))" \
  --analyze \
  --output-format stats
```

**Output includes**:
- Tensor count
- Node count
- Graph depth
- Estimated FLOPs
- Memory usage

**Use case**: Identify performance bottlenecks

---

### Recipe 22: Domain Size Tuning

```bash
# Start with small domains
tensorlogic "FORALL x IN D. p(x)" --domains D:10 --analyze

# Scale up and compare
tensorlogic "FORALL x IN D. p(x)" --domains D:100 --analyze
tensorlogic "FORALL x IN D. p(x)" --domains D:1000 --analyze
```

**Use case**: Find optimal domain sizes

---

### Recipe 23: Strategy Comparison

```bash
#!/bin/bash
EXPR="p AND q AND r"

for strategy in soft_differentiable hard_boolean fuzzy_godel fuzzy_product; do
  echo "=== $strategy ==="
  tensorlogic "$EXPR" \
    --strategy $strategy \
    --output-format stats \
    --analyze
  echo
done
```

**Use case**: Choose best strategy for your use case

---

## Troubleshooting

### Recipe 24: Debug Compilation Issues

```bash
tensorlogic "problematic(expression)" \
  --debug \
  --validate \
  2> debug.log
```

**Use case**: Diagnose compilation failures

---

### Recipe 25: Validate Expression Syntax

```bash
# Use convert to check syntax
tensorlogic convert "your(expression)" \
  --from expr \
  --to json \
  --pretty
```

**Use case**: Syntax validation without full compilation

---

### Recipe 26: Check Configuration

```bash
tensorlogic config show
tensorlogic config path
```

**Use case**: Verify settings

---

## Advanced Patterns

### Recipe 27: Multi-Rule Compilation

```bash
#!/bin/bash
# Compile multiple rules and combine

rules=(
  "knows(x,y) AND knows(y,z) -> transitive(x,z)"
  "friend(x,y) -> friend(y,x)"
  "person(x) -> mortal(x)"
)

for i in "${!rules[@]}"; do
  tensorlogic "${rules[$i]}" \
    --output-format json \
    --output "rule_$i.json"
done

# Combine with jq
jq -s '.' rule_*.json > combined_rules.json
```

---

### Recipe 28: Performance Benchmarking

```bash
#!/bin/bash
# Benchmark different strategies

time_strategy() {
  local strategy=$1
  local expr=$2
  time tensorlogic "$expr" \
    --strategy "$strategy" \
    --quiet \
    --output-format stats > /dev/null
}

EXPR="FORALL x IN D. (p(x) AND q(x)) -> r(x)"

echo "Benchmarking strategies:"
for strategy in soft_differentiable hard_boolean fuzzy_godel; do
  echo -n "$strategy: "
  time_strategy "$strategy" "$EXPR"
done
```

---

### Recipe 29: Rule Library Management

```bash
# Create rule library structure
mkdir -p rules/{social,access,temporal}

# Social network rules
cat > rules/social/friendship.tl << 'EOF'
FORALL x IN Person. FORALL y IN Person.
  friend(x, y) -> friend(y, x)
EOF

# Compile library
find rules -name "*.tl" -exec tensorlogic {} \
  --output-format json \
  --output {}.json \
  --validate \;
```

---

### Recipe 30: Error Analysis Pipeline

```bash
#!/bin/bash
# Analyze compilation errors

ERROR_LOG="compilation_errors.log"
> "$ERROR_LOG"

for rule_file in rules/*.tl; do
  if ! tensorlogic "$rule_file" --validate --quiet 2>> "$ERROR_LOG"; then
    echo "ERROR: $rule_file" | tee -a "$ERROR_LOG"
  fi
done

# Analyze errors
echo "Error Summary:"
grep -c "ERROR:" "$ERROR_LOG"
grep "ERROR:" "$ERROR_LOG" | sort | uniq -c
```

---

## Best Practices

### 1. **Always Validate in Production**

```bash
tensorlogic "$expr" --validate --output-format json
```

### 2. **Use Configuration Files for Consistency**

```bash
# .tensorlogicrc
strategy = "soft_differentiable"
validate = true

[domains]
Person = 100
Event = 1000
```

### 3. **Document Domain Sizes**

```bash
# rules_config.txt
Person: 100 entities
City: 50 entities
Event: 1000 entities

tensorlogic "$expr" \
  --domains Person:100 \
  --domains City:50 \
  --domains Event:1000
```

### 4. **Version Control Compiled Graphs**

```bash
# Compile and commit
tensorlogic "important(rule)" \
  --output-format json \
  --output rules/v1/important.json

git add rules/v1/important.json
git commit -m "Add important rule v1"
```

### 5. **Use Quiet Mode in Scripts**

```bash
if tensorlogic "$expr" --validate --quiet; then
  echo "✓ Valid"
else
  echo "✗ Invalid"
  exit 1
fi
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Basic compilation | `tensorlogic "expr"` |
| With validation | `tensorlogic "expr" --validate` |
| Analyze complexity | `tensorlogic "expr" --analyze` |
| Save to file | `tensorlogic "expr" -o file.json` |
| Change strategy | `tensorlogic "expr" -s fuzzy_godel` |
| Convert formats | `tensorlogic convert "expr" -f expr -t json` |
| Batch process | `tensorlogic batch file.txt` |
| Watch mode | `tensorlogic watch file.tl` |
| Interactive REPL | `tensorlogic repl` |
| Generate viz | `tensorlogic "expr" -F dot \| dot -Tpng -o graph.png` |

---

## Resources

- **Main Documentation**: [README.md](../README.md)
- **Project Guide**: [CLAUDE.md](../../../CLAUDE.md)
- **API Reference**: https://docs.rs/tensorlogic-cli
- **Issue Tracker**: https://github.com/cool-japan/tensorlogic/issues

---

**Happy Compiling!** 🎉
