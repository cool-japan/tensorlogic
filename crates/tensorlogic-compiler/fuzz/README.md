# Fuzzing TensorLogic Compiler

This directory contains fuzz targets for testing the robustness of the TensorLogic compiler.

## Prerequisites

Fuzzing requires **Rust nightly** and **cargo-fuzz**:

```bash
# Install nightly toolchain
rustup install nightly

# Install cargo-fuzz
cargo install cargo-fuzz
```

## Available Fuzz Targets

### 1. `fuzz_compile_expression`

Tests compilation of arbitrary TLExpr expressions with various operations:
- Predicates (unary, binary, n-ary)
- Logical operations (AND, OR, NOT, Implication)
- Quantifiers (EXISTS, FORALL)
- Arithmetic operations (Add, Multiply)
- Comparisons (LessThan, Equal, etc.)
- Conditionals (If-then-else)

**Run:**
```bash
cargo +nightly fuzz run fuzz_compile_expression
```

### 2. `fuzz_type_checking`

Tests type checking with random predicate signatures and domain assignments:
- Random domain creation
- Random predicate signatures
- Type inference and validation
- Type mismatch detection

**Run:**
```bash
cargo +nightly fuzz run fuzz_type_checking
```

### 3. `fuzz_quantifiers`

Tests quantifier handling and scope analysis:
- Nested quantifiers (up to depth 10)
- Mixed EXISTS/FORALL combinations
- Free and bound variable detection
- Scope validation

**Run:**
```bash
cargo +nightly fuzz run fuzz_quantifiers
```

### 4. `fuzz_optimizations`

Tests all optimization passes:
- Negation optimization (double negation, De Morgan's laws)
- Common Subexpression Elimination (CSE)
- Dead Code Elimination (DCE)
- Einsum optimization
- Identity simplification

**Run:**
```bash
cargo +nightly fuzz run fuzz_optimizations
```

## Running All Fuzz Targets

You can run all fuzz targets sequentially:

```bash
for target in fuzz_compile_expression fuzz_type_checking fuzz_quantifiers fuzz_optimizations; do
  echo "Running $target..."
  cargo +nightly fuzz run $target -- -max_total_time=60
done
```

## Continuous Fuzzing

For long-running fuzz sessions:

```bash
# Run for 1 hour
cargo +nightly fuzz run fuzz_compile_expression -- -max_total_time=3600

# Run with parallelization (using 4 jobs)
cargo +nightly fuzz run fuzz_compile_expression -- -jobs=4 -workers=4

# Run with maximum memory limit
cargo +nightly fuzz run fuzz_compile_expression -- -rss_limit_mb=2048
```

## Analyzing Crashes

If a crash is found, the input will be saved to `fuzz/artifacts/<target_name>/`:

```bash
# Reproduce a specific crash
cargo +nightly fuzz run fuzz_compile_expression fuzz/artifacts/fuzz_compile_expression/crash-...

# Minimize a crash case
cargo +nightly fuzz cmin fuzz_compile_expression fuzz/artifacts/fuzz_compile_expression/

# Get coverage information
cargo +nightly fuzz coverage fuzz_compile_expression
```

## Coverage Report

Generate coverage report:

```bash
cargo +nightly fuzz coverage fuzz_compile_expression
cargo cov -- show target/aarch64-apple-darwin/coverage/aarch64-apple-darwin/release/fuzz_compile_expression \
  --format=html -instr-profile=coverage/fuzz_compile_expression/coverage.profdata \
  > coverage.html
```

## Tips for Effective Fuzzing

1. **Start with shorter runs** to verify everything works
2. **Use parallelization** for faster corpus generation
3. **Monitor memory usage** - some complex expressions can consume significant memory
4. **Collect corpus** over time for better coverage
5. **Run overnight** for comprehensive testing

## Integration with CI

For CI integration, run with time limits:

```bash
# Quick smoke test (1 minute per target)
for target in fuzz_compile_expression fuzz_type_checking fuzz_quantifiers fuzz_optimizations; do
  cargo +nightly fuzz run $target -- -max_total_time=60 -max_len=200 || exit 1
done
```

## Notes

- Fuzzing uses **Address Sanitizer (ASAN)** by default to detect memory errors
- Each fuzz target has input size limits to prevent timeouts
- Complex expressions are limited to reasonable nesting depth to avoid stack overflow
- The fuzzer will automatically generate increasingly complex test cases over time

## Troubleshooting

**"nightly required" errors:**
- Make sure you're using `cargo +nightly fuzz` (note the `+nightly`)

**Timeout errors:**
- Reduce input complexity or increase timeout: `--max_total_time=<seconds>`

**Out of memory:**
- Reduce RSS limit: `-rss_limit_mb=<megabytes>`

**No crashes found:**
- This is good! It means the compiler is robust
- Try running longer or with different seeds

## References

- [cargo-fuzz documentation](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer options](https://llvm.org/docs/LibFuzzer.html#options)
