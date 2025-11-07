# Contributing to Tensorlogic

Thank you for your interest in contributing to Tensorlogic! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to the COOLJAPAN ecosystem's principles of technical excellence, collaboration, and respectful communication.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/tensorlogic.git
   cd tensorlogic
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/cool-japan/tensorlogic.git
   ```

## Development Workflow

### Building the Project

```bash
# Build all crates
cargo build

# Build with specific backend features
cargo build -p tensorlogic-scirs-backend --features simd

# Run examples
cargo run --example 00_minimal_rule
```

### Testing

We use `cargo-nextest` for faster test execution:

```bash
# Install cargo-nextest if you haven't already
cargo install cargo-nextest

# Run all tests
cargo nextest run --no-fail-fast

# Run tests for a specific crate
cargo nextest run -p tensorlogic-compiler
```

### Code Quality

Before submitting a pull request, ensure your code passes all quality checks:

```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Run clippy (MUST have zero warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Security audit
cargo audit
```

**CRITICAL**: Code MUST compile without ANY warnings. This is strictly enforced.

### File Size Limit

Single source files should not exceed **2000 lines**. If a file grows beyond this limit:
- Use module decomposition to split functionality
- Consider using the SplitRS tool: `splitrs --help`

### Naming Conventions

- Variables and functions: `snake_case`
- Types and traits: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

## SciRS2 Integration Policy

**CRITICAL**: Tensorlogic MUST use SciRS2 as its tensor execution foundation.

### Forbidden Dependencies

❌ **NEVER** import these directly:
```rust
use ndarray::Array2;        // Wrong
use rand::thread_rng;       // Wrong
use num_complex::Complex64; // Wrong
```

✅ **ALWAYS** use SciRS2 equivalents:
```rust
use scirs2_core::ndarray::{Array, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::complex::Complex64;
use scirs2_core::array;
use scirs2_autograd::Variable;
use scirs2_linalg::einsum;
```

See [SCIRS2_INTEGRATION_POLICY.md](SCIRS2_INTEGRATION_POLICY.md) for complete details.

## Workspace Dependencies

Use `workspace = true` in Cargo.toml for shared dependencies. Add new shared dependencies to the workspace root `Cargo.toml`:

```toml
[workspace.dependencies]
new-crate = "1.0"
```

Then reference in individual crate `Cargo.toml`:

```toml
[dependencies]
new-crate.workspace = true
```

## Submitting Changes

### Branch Naming

- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

### Commit Messages

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain the problem this commit solves and why you chose
this particular solution.

Refs: #issue-number
```

### Pull Request Process

1. Update your fork with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

3. Make your changes, following the guidelines above

4. Run all quality checks:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo nextest run --no-fail-fast
   ```

5. Commit your changes with clear messages

6. Push to your fork:
   ```bash
   git push origin feature/my-feature
   ```

7. Open a Pull Request on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Test results
   - Any breaking changes noted

### Pull Request Review

- PRs require approval from at least one maintainer
- Address all review comments
- Keep PRs focused and reasonably sized
- Update documentation as needed

## Testing Guidelines

### Unit Tests

Place unit tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Integration Tests

Place integration tests in the `tests/` directory.

### Test Coverage

Aim for high coverage on:
- Compiler logic
- Inference traits
- Core IR operations

## Documentation

- Add rustdoc comments to public APIs
- Update README.md for significant features
- Update TODO.md for roadmap changes
- Create examples for new features

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues and documentation first
- Join discussions in pull requests

## License

By contributing to Tensorlogic, you agree that your contributions will be licensed under the Apache-2.0 license.

Thank you for contributing to Tensorlogic!
