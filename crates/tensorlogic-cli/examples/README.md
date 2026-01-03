# TensorLogic CLI Examples

This directory contains practical examples demonstrating various use cases of the TensorLogic CLI.

## Example Files

### 1. Social Network Analysis (`social_network.tl`)

Demonstrates reasoning about social relationships:
- Transitive friendship inference
- Symmetric friend relationships
- Mutual friend detection

**Run**:
```bash
tensorlogic social_network.tl \
  --domains Person:100 \
  --validate \
  --analyze
```

**Use cases**: Social media platforms, network analysis, friend recommendations

---

### 2. Access Control (`access_control.tl`)

Role-based access control policies:
- Admin privileges
- Resource ownership
- Group-based sharing
- Permission hierarchies

**Run**:
```bash
tensorlogic access_control.tl \
  --domains User:1000 --domains Resource:5000 --domains Group:100 \
  --strategy hard_boolean \
  --validate
```

**Use cases**: Authorization systems, security policies, enterprise access management

---

### 3. Recommendation System (`recommendation.tl`)

Content and collaborative filtering logic:
- Item similarity recommendations
- Collaborative filtering
- Category-based suggestions
- Trending content

**Run**:
```bash
tensorlogic recommendation.tl \
  --domains User:10000 --domains Item:50000 \
  --strategy probabilistic \
  --output-format stats \
  --analyze
```

**Use cases**: E-commerce, content platforms, personalization engines

---

### 4. Data Validation (`data_validation.tl`)

Quality checks and constraints:
- Age range validation
- Email requirements
- Name length constraints
- Phone format validation

**Run**:
```bash
tensorlogic data_validation.tl \
  --domains Record:1000000 --domains Email:1000000 \
  --strategy hard_boolean \
  --validate
```

**Use cases**: Data quality, ETL pipelines, form validation

---

### 5. Graph Analysis (`graph_analysis.tl`)

Graph algorithms in logic:
- Reachability
- Cycle detection
- Common neighbors
- Triangle detection

**Run**:
```bash
tensorlogic graph_analysis.tl \
  --domains Node:1000 \
  --strategy fuzzy_product \
  --output-format dot | dot -Tpng -o graph.png
```

**Use cases**: Network analysis, dependency graphs, pathfinding

---

## Batch Processing

Process all examples at once:

```bash
tensorlogic batch *.tl
```

## Visualization

Generate DOT graphs for any example:

```bash
tensorlogic social_network.tl \
  --domains Person:50 \
  --output-format dot > social.dot

dot -Tpng social.dot -o social.png
dot -Tsvg social.dot -o social.svg
```

## Format Conversion

Convert examples to JSON:

```bash
for file in *.tl; do
  tensorlogic convert "$file" \
    --from expr \
    --to json \
    --output "${file%.tl}.json"
done
```

## Performance Analysis

Compare strategies on an example:

```bash
#!/bin/bash
FILE="social_network.tl"

for strategy in soft_differentiable hard_boolean fuzzy_godel fuzzy_product; do
  echo "=== $strategy ==="
  tensorlogic "$FILE" \
    --domains Person:100 \
    --strategy "$strategy" \
    --output-format stats \
    --analyze
  echo
done
```

## Integration with Build Systems

### Makefile Example

```makefile
.PHONY: all validate visualize clean

TL_FILES := $(wildcard *.tl)
JSON_FILES := $(TL_FILES:.tl=.json)
DOT_FILES := $(TL_FILES:.tl=.dot)
PNG_FILES := $(TL_FILES:.tl=.png)

all: validate $(JSON_FILES)

validate:
	tensorlogic batch $(TL_FILES)

%.json: %.tl
	tensorlogic convert $< --from expr --to json --output $@

%.dot: %.tl
	tensorlogic $< --output-format dot --output $@

%.png: %.dot
	dot -Tpng $< -o $@

visualize: $(PNG_FILES)

clean:
	rm -f $(JSON_FILES) $(DOT_FILES) $(PNG_FILES)
```

## Advanced Examples

### Custom Domains

```bash
tensorlogic social_network.tl \
  --domains Person:1000 \
  --domains City:100 \
  --domains Event:500
```

### Multiple Output Formats

```bash
# Save JSON and generate visualization
tensorlogic social_network.tl \
  --domains Person:100 \
  --output-format json \
  --output social.json

tensorlogic social_network.tl \
  --domains Person:100 \
  --output-format dot | dot -Tpng -o social.png
```

### Watch Mode for Development

```bash
# Auto-recompile on file changes
tensorlogic watch social_network.tl
```

## Tips

1. **Start Small**: Begin with small domain sizes to test logic, then scale up
2. **Validate Early**: Always use `--validate` during development
3. **Analyze Complexity**: Use `--analyze` to understand computational costs
4. **Choose Strategy**: Select the right strategy for your use case
5. **Use Quiet Mode**: Add `--quiet` for cleaner output in scripts

## Further Reading

- [COOKBOOK.md](../docs/COOKBOOK.md) - Comprehensive recipes
- [README.md](../README.md) - Full CLI documentation
- [tensorlogic.1](../docs/tensorlogic.1) - Man page

---

**Happy Experimenting!** 🚀
