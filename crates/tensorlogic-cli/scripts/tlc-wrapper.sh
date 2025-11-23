#!/usr/bin/env bash
# TensorLogic CLI Wrapper Script
# Provides convenient aliases and helper functions

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for colored output
error() { echo -e "${RED}✗${NC} $*" >&2; }
success() { echo -e "${GREEN}✓${NC} $*"; }
info() { echo -e "${BLUE}→${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

# Check if tensorlogic is installed
if ! command -v tensorlogic &> /dev/null; then
    error "tensorlogic CLI not found. Please install it first:"
    echo "  cargo install tensorlogic-cli"
    exit 1
fi

# Show help
show_help() {
    cat << EOF
TensorLogic CLI Wrapper - Convenient shortcuts and workflows

USAGE:
    tlc-wrapper.sh COMMAND [OPTIONS]

COMMANDS:
    compile FILE [STRATEGY]      Compile a rule file
    validate FILES...            Validate multiple rule files
    visualize FILE [OUTPUT]      Generate visualization
    compare FILE STRATEGIES...   Compare compilation strategies
    benchmark FILE [ITERS]       Benchmark compilation
    watch FILE                   Watch file for changes
    repl                        Start interactive REPL
    init [DIR]                  Initialize project structure
    help                        Show this help

EXAMPLES:
    # Compile a rule file
    tlc-wrapper.sh compile rules/policy.tl soft_differentiable

    # Validate all rules
    tlc-wrapper.sh validate rules/*.tl

    # Generate visualization
    tlc-wrapper.sh visualize rules/policy.tl policy.png

    # Compare strategies
    tlc-wrapper.sh compare rules/policy.tl soft_differentiable hard_boolean fuzzy_godel

    # Benchmark
    tlc-wrapper.sh benchmark rules/policy.tl 100

    # Initialize project
    tlc-wrapper.sh init my-project

EOF
}

# Compile rule file
cmd_compile() {
    local file="${1:-}"
    local strategy="${2:-soft_differentiable}"

    if [[ -z "$file" ]]; then
        error "No file specified"
        echo "Usage: tlc-wrapper.sh compile FILE [STRATEGY]"
        exit 1
    fi

    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        exit 1
    fi

    info "Compiling $file with strategy: $strategy"
    tensorlogic "$file" \
        --strategy "$strategy" \
        --validate \
        --analyze
}

# Validate multiple files
cmd_validate() {
    local files=("$@")
    local passed=0
    local failed=0

    if [[ ${#files[@]} -eq 0 ]]; then
        error "No files specified"
        echo "Usage: tlc-wrapper.sh validate FILES..."
        exit 1
    fi

    info "Validating ${#files[@]} file(s)..."
    echo

    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            warn "Skipping non-existent file: $file"
            continue
        fi

        echo -n "  $file ... "
        if tensorlogic "$file" --validate --quiet 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
            ((passed++))
        else
            echo -e "${RED}✗${NC}"
            ((failed++))
        fi
    done

    echo
    echo "Results: $passed passed, $failed failed"

    [[ $failed -eq 0 ]]
}

# Generate visualization
cmd_visualize() {
    local file="${1:-}"
    local output="${2:-}"

    if [[ -z "$file" ]]; then
        error "No file specified"
        echo "Usage: tlc-wrapper.sh visualize FILE [OUTPUT]"
        exit 1
    fi

    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        exit 1
    fi

    # Determine output filename
    if [[ -z "$output" ]]; then
        output="${file%.tl}.png"
    fi

    local dot_file="${output%.png}.dot"

    info "Generating visualization for $file"

    # Generate DOT file
    tensorlogic "$file" --output-format dot > "$dot_file"

    # Check if graphviz is installed
    if command -v dot &> /dev/null; then
        # Generate PNG
        dot -Tpng "$dot_file" -o "$output"
        success "Visualization saved to: $output"
    else
        warn "Graphviz not installed. DOT file saved to: $dot_file"
        echo "  Install graphviz to generate PNG: brew install graphviz"
    fi
}

# Compare compilation strategies
cmd_compare() {
    local file="${1:-}"
    shift
    local strategies=("$@")

    if [[ -z "$file" ]]; then
        error "No file specified"
        echo "Usage: tlc-wrapper.sh compare FILE STRATEGIES..."
        exit 1
    fi

    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        exit 1
    fi

    if [[ ${#strategies[@]} -eq 0 ]]; then
        # Default strategies to compare
        strategies=(soft_differentiable hard_boolean fuzzy_godel fuzzy_product)
    fi

    info "Comparing ${#strategies[@]} strategies for: $file"
    echo

    for strategy in "${strategies[@]}"; do
        echo -e "${BLUE}=== $strategy ===${NC}"
        tensorlogic "$file" \
            --strategy "$strategy" \
            --output-format stats \
            2>/dev/null || warn "Failed to compile with $strategy"
        echo
    done
}

# Benchmark file
cmd_benchmark() {
    local file="${1:-}"
    local iterations="${2:-100}"

    if [[ -z "$file" ]]; then
        error "No file specified"
        echo "Usage: tlc-wrapper.sh benchmark FILE [ITERATIONS]"
        exit 1
    fi

    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        exit 1
    fi

    info "Benchmarking $file ($iterations iterations)"

    tensorlogic benchmark "$file" \
        --input-format expr \
        --iterations "$iterations" \
        --verbose
}

# Watch file
cmd_watch() {
    local file="${1:-}"

    if [[ -z "$file" ]]; then
        error "No file specified"
        echo "Usage: tlc-wrapper.sh watch FILE"
        exit 1
    fi

    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        exit 1
    fi

    info "Watching $file for changes (Ctrl+C to stop)"
    tensorlogic watch "$file"
}

# Start REPL
cmd_repl() {
    info "Starting TensorLogic REPL"
    tensorlogic repl
}

# Initialize project structure
cmd_init() {
    local dir="${1:-.}"

    info "Initializing TensorLogic project in: $dir"

    mkdir -p "$dir"/{rules,docs,tests}

    # Create example rule
    cat > "$dir/rules/example.tl" << 'EOF'
# Example TensorLogic Rules

# Simple predicate
knows(alice, bob)

# Logical operations
knows(x, y) AND likes(y, z)

# Quantifiers
EXISTS x IN Person. knows(x, alice)

# Implication
knows(x, y) -> likes(x, y)
EOF

    # Create config
    cat > "$dir/.tensorlogicrc" << 'EOF'
# TensorLogic Configuration

strategy = "soft_differentiable"
colored = true
validate = false

[domains]
Person = 100

[repl]
prompt = "tensorlogic> "
max_history = 1000

[watch]
debounce_ms = 500
clear_screen = true
EOF

    # Create README
    cat > "$dir/README.md" << 'EOF'
# TensorLogic Project

This project uses TensorLogic for logic-to-tensor compilation.

## Structure

- `rules/`: Logic rule files (.tl)
- `docs/`: Documentation and visualizations
- `tests/`: Test files

## Usage

```bash
# Compile a rule
tensorlogic rules/example.tl

# Validate all rules
tensorlogic batch rules/*.tl

# Visualize a rule
tensorlogic rules/example.tl --output-format dot | dot -Tpng -o docs/example.png
```

## Resources

- [TensorLogic Documentation](https://github.com/cool-japan/tensorlogic)
- [CLI Tutorial](https://github.com/cool-japan/tensorlogic/blob/main/crates/tensorlogic-cli/TUTORIAL.md)
EOF

    success "Project initialized in: $dir"
    echo
    echo "Structure created:"
    echo "  rules/example.tl       - Example rule file"
    echo "  .tensorlogicrc        - Configuration file"
    echo "  README.md             - Project documentation"
    echo
    info "Next steps:"
    echo "  cd $dir"
    echo "  tensorlogic rules/example.tl"
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        compile)     cmd_compile "$@" ;;
        validate)    cmd_validate "$@" ;;
        visualize)   cmd_visualize "$@" ;;
        compare)     cmd_compare "$@" ;;
        benchmark)   cmd_benchmark "$@" ;;
        watch)       cmd_watch "$@" ;;
        repl)        cmd_repl "$@" ;;
        init)        cmd_init "$@" ;;
        help|--help|-h) show_help ;;
        *)
            error "Unknown command: $command"
            echo "Run 'tlc-wrapper.sh help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
