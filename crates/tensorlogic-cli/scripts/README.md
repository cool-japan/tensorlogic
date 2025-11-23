# TensorLogic CLI Scripts

Helper scripts and utilities for working with TensorLogic CLI.

## Available Scripts

### tlc-wrapper.sh

Convenient wrapper script with common workflows and shortcuts.

**Installation:**
```bash
# Copy to your PATH
cp scripts/tlc-wrapper.sh /usr/local/bin/tlc
chmod +x /usr/local/bin/tlc

# Or create an alias
echo 'alias tlc="$(pwd)/scripts/tlc-wrapper.sh"' >> ~/.bashrc
source ~/.bashrc
```

**Usage:**
```bash
# Compile a rule file
tlc compile rules/policy.tl soft_differentiable

# Validate multiple files
tlc validate rules/*.tl

# Generate visualization
tlc visualize rules/policy.tl output.png

# Compare strategies
tlc compare rules/policy.tl soft_differentiable hard_boolean fuzzy_godel

# Benchmark
tlc benchmark rules/policy.tl 100

# Watch for changes
tlc watch rules/policy.tl

# Start REPL
tlc repl

# Initialize new project
tlc init my-project
```

## Integration Examples

### Git Hooks

**pre-commit** - Validate rules before commit:
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Validating TensorLogic rules..."
if scripts/tlc-wrapper.sh validate rules/*.tl; then
    echo "✓ All rules valid"
    exit 0
else
    echo "✗ Validation failed"
    exit 1
fi
```

### CI/CD Integration

**GitHub Actions:**
```yaml
name: Validate Rules

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Install TensorLogic CLI
        run: cargo install tensorlogic-cli
      - name: Validate rules
        run: ./scripts/tlc-wrapper.sh validate rules/*.tl
```

**GitLab CI:**
```yaml
validate-rules:
  stage: test
  script:
    - cargo install tensorlogic-cli
    - ./scripts/tlc-wrapper.sh validate rules/*.tl
```

### Makefile Integration

```makefile
.PHONY: validate visualize clean

TL_FILES := $(wildcard rules/*.tl)
GRAPHS := $(TL_FILES:.tl=.png)

validate:
	@scripts/tlc-wrapper.sh validate $(TL_FILES)

visualize: $(GRAPHS)

%.png: %.tl
	@scripts/tlc-wrapper.sh visualize $< $@

clean:
	rm -f rules/*.dot rules/*.png

watch:
	@while true; do \
		make validate; \
		sleep 2; \
	done
```

## Custom Scripts

You can create custom scripts using the TensorLogic CLI:

### Example: Rule Diff Tool

```bash
#!/bin/bash
# Compare two versions of a rule

RULE1="$1"
RULE2="$2"

tensorlogic "$RULE1" --output-format json > /tmp/rule1.json
tensorlogic "$RULE2" --output-format json > /tmp/rule2.json

diff -u /tmp/rule1.json /tmp/rule2.json
```

### Example: Batch Converter

```bash
#!/bin/bash
# Convert all .tl files to JSON

for file in rules/*.tl; do
    output="${file%.tl}.json"
    echo "Converting $file -> $output"
    tensorlogic "$file" --output-format json --output "$output"
done
```

### Example: Performance Report

```bash
#!/bin/bash
# Generate performance report for all rules

REPORT="performance_report.md"

echo "# TensorLogic Performance Report" > "$REPORT"
echo "Generated: $(date)" >> "$REPORT"
echo >> "$REPORT"

for file in rules/*.tl; do
    echo "## $(basename $file)" >> "$REPORT"
    echo '```' >> "$REPORT"
    tensorlogic "$file" --output-format stats 2>&1 >> "$REPORT"
    echo '```' >> "$REPORT"
    echo >> "$REPORT"
done

echo "Report saved to: $REPORT"
```

## Shell Functions

Add these to your `.bashrc` or `.zshrc`:

```bash
# Quick compile
tlc-compile() {
    tensorlogic "$1" --validate --analyze
}

# Quick validate
tlc-validate() {
    tensorlogic "$1" --validate --quiet && echo "✓ Valid" || echo "✗ Invalid"
}

# Quick stats
tlc-stats() {
    tensorlogic "$1" --output-format stats
}

# Quick visualize
tlc-viz() {
    local input="$1"
    local output="${input%.tl}.png"
    tensorlogic "$input" --output-format dot | dot -Tpng -o "$output"
    echo "Saved to: $output"
}
```

## Tips

1. **Use aliases** for frequently used commands
2. **Combine with jq** for JSON processing:
   ```bash
   tensorlogic rule.tl --output-format json | jq '.tensors | length'
   ```

3. **Use watch** for live development:
   ```bash
   watch -n 1 'tensorlogic rule.tl --output-format stats'
   ```

4. **Pipeline with other tools**:
   ```bash
   tensorlogic rule.tl --output-format dot | \
     dot -Tsvg | \
     svg2png > output.png
   ```

## Contributing

Found a useful script pattern? Please contribute it back to the project!

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.
