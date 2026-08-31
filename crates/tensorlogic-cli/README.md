# TensorLogic CLI

**Command-line interface for TensorLogic compilation**

[![Crates.io](https://img.shields.io/crates/v/tensorlogic-cli.svg)](https://crates.io/crates/tensorlogic-cli)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/status-stable-success)](#)

**Version**: 0.1.2 | **Status**: Stable | **Last Updated**: 2026-08-30

A comprehensive command-line tool for compiling logical expressions to tensor graphs using TensorLogic.

## Features

- **Multiple Modes**: Interactive REPL, batch processing, watch mode
- **Rich Input Formats**: Expression strings, JSON, YAML, stdin
- **Multiple Output Formats**: Graph, DOT, JSON, statistics
- **6 Compilation Strategies**: Differentiable, Boolean, fuzzy logic variants
- **Colored Output**: Terminal output with status indicators
- **Graph Analysis**: Complexity metrics, FLOP estimation, memory analysis
- **Configuration Files**: Persistent settings via `.tensorlogicrc`
- **File Watching**: Auto-recompilation on file changes
- **Batch Processing**: Process multiple expressions with progress bars (parallel via rayon)
- **Shell Completion**: Bash, Zsh, Fish, PowerShell support
- **Enhanced Parser**: Arithmetic, comparisons, conditionals, Unicode operators
- **Visualization** (v0.1.6): `DotExporter` for Graphviz DOT output, `AsciiRenderer` for terminal graph display
- **Library Mode**: Use CLI functionality programmatically from Rust
- **Macro System**: Define and reuse parameterized logical patterns
- **FFI Bindings**: C/C++ and Python integration via FFI
- **Persistent Cache**: LRU disk cache with analytics and compression

## Installation

### From crates.io

```bash
cargo install tensorlogic-cli
```

### From source

```bash
git clone https://github.com/cool-japan/tensorlogic.git
cd tensorlogic
cargo install --path crates/tensorlogic-cli
```

### Development build

```bash
cargo build -p tensorlogic-cli --release
# Binary at target/release/tensorlogic-cli
```

## Quick Start

### Basic Compilation

```bash
tensorlogic-cli "knows(x, y)"
```

### Interactive REPL

```bash
tensorlogic-cli repl
```

```
TensorLogic Interactive REPL
Type '.help' for available commands, '.exit' to quit

tensorlogic> knows(x, y) AND likes(y, z)
✓ Compilation successful
  3 tensors, 3 nodes, depth 2
```

### Batch Processing

```bash
# expressions.txt contains one expression per line
tensorlogic-cli batch expressions.txt
```

### Watch Mode

```bash
tensorlogic-cli watch my_expression.tl
```

## Usage

### Command Structure

```bash
tensorlogic-cli [OPTIONS] <INPUT>
tensorlogic-cli <SUBCOMMAND>
```

### Main Options

| Option | Description |
|--------|-------------|
| `-f, --input-format <FORMAT>` | Input format: `expr`, `json`, `yaml` (default: `expr`) |
| `-o, --output <FILE>` | Output file (default: stdout) |
| `-F, --output-format <FORMAT>` | Output format: `graph`, `dot`, `json`, `stats` (default: `graph`) |
| `-s, --strategy <STRATEGY>` | Compilation strategy |
| `-d, --domain <NAME:SIZE>` | Define domain (can be repeated) |
| `--validate` | Enable graph validation |
| `--debug` | Enable debug output |
| `-a, --analyze` | Show graph analysis metrics |
| `-q, --quiet` | Quiet mode (minimal output) |
| `--no-color` | Disable colored output |
| `--no-config` | Don't load configuration file |

### Subcommands

#### `repl` - Interactive REPL

Start an interactive Read-Eval-Print Loop for exploring TensorLogic.

```bash
tensorlogic-cli repl
```

**REPL Commands:**
- `.help` - Show help
- `.exit` - Exit REPL
- `.clear` - Clear screen
- `.context` - Show compiler context
- `.domain <name> <size>` - Add domain
- `.strategy [name]` - Show/set strategy
- `.validate` - Toggle validation
- `.debug` - Toggle debug mode
- `.history` - Show command history

#### `batch` - Batch Processing

Process multiple expressions from files (one expression per line).

```bash
tensorlogic-cli batch file1.txt file2.txt
```

Features:
- Progress bar with elapsed time
- Summary statistics (successes/failures)
- Line-by-line error reporting
- Comments support (`#` prefix)

#### `watch` - File Watching

Watch a file and recompile on changes.

```bash
tensorlogic-cli watch expression.tl
```

Features:
- Auto-recompilation on file save
- Debounced updates (configurable)
- Clear screen on reload (configurable)
- Timestamp display

#### `completion` - Shell Completion

Generate shell completion scripts.

```bash
# Bash
tensorlogic-cli completion bash > /etc/bash_completion.d/tensorlogic-cli

# Zsh
tensorlogic-cli completion zsh > ~/.zsh/completion/_tensorlogic-cli

# Fish
tensorlogic-cli completion fish > ~/.config/fish/completions/tensorlogic-cli.fish

# PowerShell
tensorlogic-cli completion powershell > tensorlogic-cli.ps1
```

#### `config` - Configuration Management

Manage configuration files.

```bash
# Show current configuration
tensorlogic-cli config show

# Show config file path
tensorlogic-cli config path

# Initialize default config
tensorlogic-cli config init

# Edit configuration
tensorlogic-cli config edit
```

## Configuration File

TensorLogic CLI supports persistent configuration via `.tensorlogicrc` (TOML format).

**Search order:**
1. `TENSORLOGIC_CONFIG` environment variable
2. `.tensorlogicrc` in current directory
3. `.tensorlogicrc` in home directory

**Example configuration:**

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
auto_save = true

# Watch mode settings
[watch]
debounce_ms = 500
clear_screen = true
show_timestamps = true
```

### Initialize Configuration

```bash
tensorlogic-cli config init
```

## Input Formats

### 1. Expression String (Default)

Direct expression input with enhanced syntax:

```bash
tensorlogic-cli "knows(x, y) AND likes(y, z)"
```

**Supported syntax:**
- **Predicates**: `pred(x, y, ...)`
- **Logical**: `AND` (`&`, `&&`, `∧`), `OR` (`|`, `||`, `∨`), `NOT` (`~`, `!`, `¬`), `IMPLIES` (`->`, `=>`, `→`)
- **Quantifiers**: `EXISTS x IN Domain. expr` (`∃`), `FORALL x IN Domain. expr` (`∀`)
- **Arithmetic**: `+`, `-`, `*` (`×`), `/` (`÷`)
- **Comparisons**: `=` (`==`), `<`, `>`, `<=` (`≤`), `>=` (`≥`), `!=` (`≠`)
- **Conditional**: `IF cond THEN x ELSE y`
- **Parentheses**: `(...)` for grouping

**Examples:**
```bash
# Basic predicate
tensorlogic-cli "person(x)"

# Logical operations
tensorlogic-cli "p(x) AND q(y) OR r(z)"

# Quantifiers
tensorlogic-cli "EXISTS x IN Person. knows(x, alice)"
tensorlogic-cli "FORALL x IN Person. likes(x, pizza)"

# Arithmetic
tensorlogic-cli "age(x) + 10"

# Comparisons
tensorlogic-cli "age(x) > 18"

# Conditional
tensorlogic-cli "IF age(x) >= 18 THEN adult(x) ELSE child(x)"

# Complex expression
tensorlogic-cli "(p(x) OR q(y)) AND (r(z) -> s(w))"
```

### 2. JSON Input

```bash
tensorlogic-cli --input-format json expression.json
```

```json
{
  "And": {
    "left": {
      "Pred": {
        "name": "knows",
        "args": [{"Var": "x"}, {"Var": "y"}]
      }
    },
    "right": {
      "Pred": {
        "name": "likes",
        "args": [{"Var": "y"}, {"Var": "z"}]
      }
    }
  }
}
```

### 3. YAML Input

```bash
tensorlogic-cli --input-format yaml expression.yaml
```

```yaml
And:
  left:
    Pred:
      name: knows
      args:
        - Var: x
        - Var: y
  right:
    Pred:
      name: likes
      args:
        - Var: y
        - Var: z
```

### 4. Stdin Input

```bash
echo '{"Pred": {"name": "test", "args": []}}' | tensorlogic-cli --input-format json -
```

## Output Formats

### 1. Graph (Default)

Human-readable graph structure:

```bash
tensorlogic-cli "knows(x, y)" --output-format graph
```

### 2. DOT Format

Generate Graphviz DOT for visualization:

```bash
tensorlogic-cli "knows(x, y)" --output-format dot > graph.dot
dot -Tpng graph.dot -o graph.png
```

### 3. JSON

Machine-readable JSON output:

```bash
tensorlogic-cli "knows(x, y)" --output-format json
```

### 4. Statistics

Graph statistics and metrics:

```bash
tensorlogic-cli "knows(x, y) AND likes(y, z)" --output-format stats
```

Output:
```
Graph Statistics:
  Tensors: 3
  Nodes: 3
  Inputs: 2
  Outputs: 1
  Depth: 2
  Avg Fanout: 1.00

Operation Breakdown:
  Einsum: 2
  ElemBinary: 1

Estimated Complexity:
  FLOPs: 6000
  Memory: 24000 bytes
```

## Compilation Strategies

Choose from 6 preset strategies:

### 1. Soft Differentiable (Default)

For neural network training with smooth gradients:

```bash
tensorlogic-cli --strategy soft_differentiable "p AND q"
```

- AND: Element-wise product
- OR: Probabilistic sum
- NOT: Complement (1 - x)

### 2. Hard Boolean

For discrete Boolean logic:

```bash
tensorlogic-cli --strategy hard_boolean "p AND q"
```

- AND: Minimum
- OR: Maximum
- NOT: Complement

### 3. Fuzzy Gödel

Gödel fuzzy logic (min/max operations):

```bash
tensorlogic-cli --strategy fuzzy_godel "p AND q"
```

### 4. Fuzzy Product

Product fuzzy logic (probabilistic):

```bash
tensorlogic-cli --strategy fuzzy_product "p AND q"
```

### 5. Fuzzy Łukasiewicz

Łukasiewicz fuzzy logic (bounded):

```bash
tensorlogic-cli --strategy fuzzy_lukasiewicz "p AND q"
```

### 6. Probabilistic

Probabilistic interpretation:

```bash
tensorlogic-cli --strategy probabilistic "p AND q"
```

## Domain Definitions

Define domains for variables:

```bash
tensorlogic-cli --domain Person:100 --domain City:50 "lives_in(x, c)"
```

Multiple domains:

```bash
tensorlogic-cli \
  --domain Person:100 \
  --domain Location:50 \
  --domain Event:20 \
  "attends(p, e) AND located_at(e, l)"
```

## Graph Analysis

Enable detailed analysis with `--analyze`:

```bash
tensorlogic-cli "complex(expression)" --analyze
```

Output includes:
- Tensor and node counts
- Graph depth (longest path)
- Average fanout (outputs per node)
- Operation breakdown (Einsum, element-wise, reduction)
- Estimated FLOPs (computational cost)
- Estimated memory usage

## Validation

Enable graph validation:

```bash
tensorlogic-cli "knows(x, y)" --validate
```

Checks:
- Free variable consistency
- Arity validation
- Type checking
- Graph structure validity

## Debug Mode

Enable detailed debug output:

```bash
tensorlogic-cli "knows(x, y)" --debug
```

Shows:
- Parsed expression (AST)
- Compiler context (domains, strategies)
- Intermediate compilation steps
- Final graph structure
- Validation results

## Examples

### Example 1: Simple Predicate

```bash
tensorlogic-cli "knows(alice, bob)"
```

### Example 2: Logical Conjunction

```bash
tensorlogic-cli "knows(x, y) AND likes(y, z)"
```

### Example 3: Quantifier

```bash
tensorlogic-cli --domain Person:100 "EXISTS x IN Person. knows(x, bob)"
```

### Example 4: Implication

```bash
tensorlogic-cli "knows(x, y) -> likes(x, y)"
```

### Example 5: Arithmetic and Comparison

```bash
tensorlogic-cli "age(x) + 10 > 30"
```

### Example 6: Conditional Expression

```bash
tensorlogic-cli "IF age(x) >= 18 THEN adult(x) ELSE child(x)"
```

### Example 7: Visualization

```bash
tensorlogic-cli "knows(x, y) AND likes(y, z)" \
  --output-format dot \
  --validate > graph.dot
dot -Tpng graph.dot -o graph.png
```

### Example 8: Complex Expression with Analysis

```bash
tensorlogic-cli \
  "FORALL x IN Person. (knows(x, y) -> likes(x, y))" \
  --domain Person:100 \
  --strategy fuzzy_godel \
  --output-format stats \
  --analyze \
  --validate
```

### Example 9: Batch Processing with Progress

```bash
cat > expressions.txt << EOF
knows(x, y)
likes(y, z)
knows(x, y) AND likes(y, z)
EXISTS x. knows(x, bob)
FORALL x. person(x) -> mortal(x)
EOF

tensorlogic-cli batch expressions.txt
```

### Example 10: Interactive REPL Session

```bash
tensorlogic-cli repl
```

```
tensorlogic> .domain Person 100
✓ Added domain 'Person' with size 100

tensorlogic> .strategy fuzzy_godel
✓ Strategy set to: fuzzy_godel

tensorlogic> EXISTS x IN Person. knows(x, alice)
✓ Compilation successful
  2 tensors, 2 nodes, depth 2

tensorlogic> .history
   1: .domain Person 100
   2: .strategy fuzzy_godel
   3: EXISTS x IN Person. knows(x, alice)

tensorlogic> .exit
```

## Integration

### With Graphviz

Generate PNG visualization:

```bash
tensorlogic-cli "knows(x, y)" --output-format dot | dot -Tpng -o graph.png
```

### With jq (JSON Processing)

Extract tensor count:

```bash
tensorlogic-cli "knows(x, y)" --output-format json | jq '.tensors | length'
```

### In Scripts

```bash
#!/bin/bash
EXPR="knows(x, y) AND likes(y, z)"

if tensorlogic-cli "$EXPR" --output-format stats --validate --quiet; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi
```

### With Make

```makefile
.PHONY: compile watch

compile:
	tensorlogic-cli expression.tl --validate --analyze

watch:
	tensorlogic-cli watch expression.tl
```

## Troubleshooting

### Command Not Found

Make sure `~/.cargo/bin` is in your PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Parsing Errors

Use `--debug` to see detailed parsing information:

```bash
tensorlogic-cli "your expression" --debug
```

### Validation Failures

Check free variables and domains:

```bash
tensorlogic-cli "EXISTS x. p(x, y)" \
  --domain Domain:10 \
  --debug \
  --validate
```

### Configuration Issues

Show current configuration:

```bash
tensorlogic-cli config show
```

Show config file path:

```bash
tensorlogic-cli config path
```

Reinitialize configuration:

```bash
tensorlogic-cli config init
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TENSORLOGIC_CONFIG` | Custom config file path |
| `EDITOR` | Editor for `config edit` command (default: `vi`) |

Example:

```bash
export TENSORLOGIC_CONFIG=~/.config/tensorlogic.toml
export EDITOR=nano
tensorlogic-cli config edit
```

## Performance Tips

1. **Use batch mode** for multiple expressions
2. **Enable validation** only when needed
3. **Use quiet mode** (`-q`) in scripts
4. **Disable colors** (`--no-color`) for log files
5. **Use analysis** (`--analyze`) to identify bottlenecks

## Development

### Building

```bash
cargo build -p tensorlogic-cli
```

### Testing

```bash
cargo nextest run -p tensorlogic-cli --all-features
# 291 tests run: 291 passed (as of 2026-06-09)
```

### Running from Source

```bash
cargo run -p tensorlogic-cli -- --help
```

### Code Structure

```
src/
├── main.rs            - Main entry point and command routing
├── cli.rs             - Clap CLI definitions
├── config.rs          - Configuration file support
├── parser.rs          - Enhanced expression parser
├── output.rs          - Colored output formatting
├── analysis.rs        - Graph metrics and analysis
├── repl.rs            - Interactive REPL mode
├── batch.rs           - Parallel batch processing (rayon)
├── watch.rs           - File watching and auto-recompilation
├── completion.rs      - Shell completion generation
├── executor.rs        - Execution engine with backend selection
├── optimize.rs        - Optimization pipeline with real passes
├── benchmark.rs       - Performance benchmarking
├── profile.rs         - Profiling with execution metrics
├── cache.rs           - LRU cache with analytics, warmup, compression
├── simplify.rs        - Expression simplification (constant folding, laws)
├── macros.rs          - Macro system with expansion engine
├── conversion.rs      - Format conversion and pretty-printing
├── snapshot.rs        - Snapshot management
├── ffi.rs             - FFI bindings for C/C++ integration
├── error_suggestions.rs - Actionable error suggestions
├── visualization.rs   - DotExporter and AsciiRenderer (v0.1.6)
└── lib.rs             - Library API and public exports
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

## See Also

- **Main README**: [README.md](../../README.md)
- **Project Guide**: [CLAUDE.md](../../CLAUDE.md)
- **Meta Crate**: [tensorlogic](../tensorlogic/README.md)
- **Compiler Documentation**: [tensorlogic-compiler](../tensorlogic-compiler/README.md)
- **IR Documentation**: [tensorlogic-ir](../tensorlogic-ir/README.md)

## Resources

- **Repository**: https://github.com/cool-japan/tensorlogic
- **Documentation**: https://docs.rs/tensorlogic-cli
- **Issues**: https://github.com/cool-japan/tensorlogic/issues

---

**Part of the COOLJAPAN Ecosystem**

For questions and support, please open an issue on [GitHub](https://github.com/cool-japan/tensorlogic/issues).
