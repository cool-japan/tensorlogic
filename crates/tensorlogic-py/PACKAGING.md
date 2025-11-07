# TensorLogic Python Bindings - Packaging Guide

**Version**: 1.0
**Last Updated**: 2025-11-04
**Status**: Complete Maturin packaging and distribution guide

This comprehensive guide covers building, packaging, and distributing the TensorLogic Python bindings using Maturin.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Building for Development](#building-for-development)
- [Building Wheels for Distribution](#building-wheels-for-distribution)
- [Cross-Platform Builds](#cross-platform-builds)
- [Publishing to PyPI](#publishing-to-pypi)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

TensorLogic uses [Maturin](https://github.com/PyO3/maturin) to build Python wheels from Rust code. Maturin handles:

- Compiling Rust code to Python extension modules
- Generating wheel files compatible with different Python versions
- Managing dependencies and features
- Cross-compilation support

The bindings use:
- **PyO3**: Rust ↔ Python interoperability
- **abi3**: Stable Python ABI (compatible with Python 3.9+)
- **maturin**: Build and packaging tool

## Prerequisites

### Required Tools

1. **Rust Toolchain** (1.70.0+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```

2. **Python** (3.9+):
   ```bash
   python3 --version  # Should be 3.9 or higher
   ```

3. **Maturin**:
   ```bash
   pip install maturin

   # Or with uv (faster):
   pip install uv
   uv pip install maturin
   ```

4. **Development Dependencies**:
   ```bash
   pip install -r crates/pytensorlogic/requirements-dev.txt
   ```

### Optional Tools

- **cargo-nextest**: Faster test runner
  ```bash
  cargo install cargo-nextest
  ```

- **twine**: For uploading to PyPI
  ```bash
  pip install twine
  ```

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/cool-japan/tensorlogic.git
cd tensorlogic
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
cd crates/pytensorlogic
pip install -r requirements-dev.txt
```

## Building for Development

### Basic Development Build

The fastest way to test changes during development:

```bash
cd crates/pytensorlogic

# Build and install in development mode (editable)
maturin develop

# Test the installation
python -c "import pytensorlogic as tl; print(tl.__version__)"
```

**What `maturin develop` does**:
- Compiles Rust code in debug mode (fast compilation)
- Installs the module directly into your current Python environment
- Changes to Python stub files (.pyi) are immediately visible
- Changes to Rust code require re-running `maturin develop`

### Development Build with Features

```bash
# Build with SIMD support
maturin develop --features simd

# Build with release optimizations (slower build, faster runtime)
maturin develop --release

# Build with specific Rust features
maturin develop --features "simd,cpu"
```

### Running Tests During Development

```bash
# Python tests
pytest tests/

# Rust tests
cargo test -p pytensorlogic

# All workspace tests
cargo test --workspace
```

### Incremental Development Workflow

```bash
# 1. Make changes to Rust code
vim src/lib.rs

# 2. Rebuild
maturin develop

# 3. Test changes
python examples/00_minimal_rule.py

# 4. Run tests
pytest tests/test_types.py -v
```

## Building Wheels for Distribution

### Release Wheel (Current Platform)

Build an optimized wheel for your current platform:

```bash
cd crates/pytensorlogic

# Build release wheel
maturin build --release

# Output: target/wheels/pytensorlogic-0.1.0-*.whl
```

### Build Options

```bash
# With SIMD acceleration
maturin build --release --features simd

# Specific Python version
maturin build --release --interpreter python3.9

# All available Python versions on system
maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12

# With maximum optimization
RUSTFLAGS="-C target-cpu=native" maturin build --release --features simd
```

### Installing Built Wheel

```bash
# Find the built wheel
ls target/wheels/

# Install it
pip install target/wheels/pytensorlogic-0.1.0-cp39-abi3-linux_x86_64.whl
```

## Cross-Platform Builds

### Linux → Multiple Platforms

Use `manylinux` Docker containers for maximum compatibility:

```bash
# Install cross (if not already installed)
cargo install cross

# Build for manylinux (widely compatible Linux wheels)
docker run --rm -v $(pwd):/io \
  ghcr.io/pyo3/maturin:latest \
  build --release -m crates/pytensorlogic/Cargo.toml

# This creates wheels compatible with most Linux distributions
```

### macOS Universal Wheels

Build wheels that work on both Intel and Apple Silicon:

```bash
# Install targets
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin

# Build universal wheel
maturin build --release \
  --target x86_64-apple-darwin \
  --target aarch64-apple-darwin \
  --universal2
```

### Windows Cross-Compilation

```bash
# From Linux, build Windows wheels:
rustup target add x86_64-pc-windows-gnu
maturin build --release --target x86_64-pc-windows-gnu

# From macOS/Linux with cross:
cross build --release --target x86_64-pc-windows-msvc
```

### Platform-Specific Wheels Matrix

| Platform | Architecture | Target Triple | Wheel Tag |
|----------|-------------|---------------|-----------|
| Linux | x86_64 | x86_64-unknown-linux-gnu | manylinux_2_17_x86_64 |
| Linux | aarch64 | aarch64-unknown-linux-gnu | manylinux_2_17_aarch64 |
| macOS | x86_64 | x86_64-apple-darwin | macosx_10_12_x86_64 |
| macOS | ARM64 | aarch64-apple-darwin | macosx_11_0_arm64 |
| Windows | x86_64 | x86_64-pc-windows-msvc | win_amd64 |

## Publishing to PyPI

### 1. Configure PyPI Credentials

```bash
# Create ~/.pypirc
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TEST-API-TOKEN-HERE
EOF

chmod 600 ~/.pypirc
```

### 2. Build Release Wheels

```bash
cd crates/pytensorlogic

# Clean previous builds
rm -rf target/wheels/*

# Build for multiple Python versions
maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12

# Verify wheels
ls -lh target/wheels/
```

### 3. Test on TestPyPI First

```bash
# Upload to TestPyPI
maturin upload --repository testpypi target/wheels/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ pytensorlogic

# Test it works
python -c "import pytensorlogic; print('Success!')"
```

### 4. Publish to PyPI

**Important**: Due to workspace path dependencies, use `--no-sdist` to skip source distribution:

```bash
# Upload to production PyPI (without source distribution)
maturin publish --no-sdist

# The --no-sdist flag is required because:
# - TensorLogic uses local path dependencies from the workspace
# - Maturin cannot package these dependencies into a source distribution
# - Binary wheels work perfectly fine without sdist
# - Users can install from source using: pip install git+https://github.com/cool-japan/tensorlogic

# Alternative: Build wheels first, then upload
maturin build --release
maturin upload target/wheels/*

# Or use twine
twine upload target/wheels/*
```

### 5. Verify Publication

```bash
# Install from PyPI
pip install pytensorlogic

# Check version
python -c "import pytensorlogic; print(pytensorlogic.__version__)"
```

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/python-wheels.yml`:

```yaml
name: Build Python Wheels

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  linux:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [x86_64, aarch64]
    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist -m crates/pytensorlogic/Cargo.toml
          sccache: 'true'
          manylinux: auto

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-linux-${{ matrix.target }}
          path: dist

  macos:
    runs-on: macos-latest
    strategy:
      matrix:
        target: [x86_64, aarch64]
    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist -m crates/pytensorlogic/Cargo.toml

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-macos-${{ matrix.target }}
          path: dist

  windows:
    runs-on: windows-latest
    strategy:
      matrix:
        target: [x64]
    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist -m crates/pytensorlogic/Cargo.toml

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-windows-${{ matrix.target }}
          path: dist

  release:
    name: Release
    runs-on: ubuntu-latest
    needs: [linux, macos, windows]
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist

      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
        with:
          command: upload
          args: --skip-existing dist/*
```

### GitLab CI Configuration

Create `.gitlab-ci.yml`:

```yaml
stages:
  - build
  - test
  - deploy

variables:
  CARGO_HOME: $CI_PROJECT_DIR/cargo

build:linux:
  stage: build
  image: ghcr.io/pyo3/maturin:latest
  script:
    - cd crates/pytensorlogic
    - maturin build --release --out dist
  artifacts:
    paths:
      - crates/pytensorlogic/target/wheels/
    expire_in: 1 week

test:
  stage: test
  image: python:3.11
  dependencies:
    - build:linux
  script:
    - pip install crates/pytensorlogic/target/wheels/*.whl
    - pip install pytest numpy
    - cd crates/pytensorlogic
    - pytest tests/

deploy:pypi:
  stage: deploy
  image: python:3.11
  dependencies:
    - build:linux
  only:
    - tags
  script:
    - pip install maturin
    - cd crates/pytensorlogic
    - maturin upload target/wheels/*
  environment:
    name: production
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "maturin: command not found"

**Problem**: Maturin not installed or not in PATH

**Solution**:
```bash
pip install --user maturin
# Add ~/.local/bin to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

#### 2. "error: could not compile `pyo3`"

**Problem**: Incompatible Rust/PyO3 version

**Solution**:
```bash
rustup update stable
cargo clean
maturin develop
```

#### 3. "ImportError: undefined symbol"

**Problem**: ABI incompatibility between Python versions

**Solution**:
```bash
# Rebuild for specific Python version
maturin develop --interpreter python3.11

# Or use abi3 (should already be configured)
# Check Cargo.toml has: pyo3 = { version = "0.20", features = ["abi3-py39"] }
```

#### 4. "No matching distribution found"

**Problem**: Wheel not available for your platform

**Solution**:
```bash
# Build from source
pip install maturin
cd crates/pytensorlogic
maturin develop
```

#### 5. Link Errors on macOS

**Problem**: Missing system libraries

**Solution**:
```bash
xcode-select --install
brew install python@3.11
```

#### 6. "wheel is not a supported wheel on this platform"

**Problem**: Platform mismatch

**Solution**:
```bash
# Check wheel compatibility
python -m wheel tags

# Rebuild for correct platform
maturin build --release --interpreter $(which python3)
```

#### 7. "StripPrefixError" when running `maturin publish`

**Problem**: Maturin panics when creating source distribution with workspace path dependencies

**Error Message**:
```
thread 'main' panicked at src/source_distribution.rs:720:14:
called `Result::unwrap()` on an `Err` value: StripPrefixError(())
```

**Solution**:
```bash
# Use --no-sdist flag to skip source distribution creation
maturin publish --no-sdist

# This is the recommended approach for packages with workspace dependencies
# Binary wheels will be published, and users can install from git if needed
```

### Debug Build Issues

```bash
# Enable verbose output
maturin develop --verbose

# Check Python configuration
python -c "import sysconfig; print(sysconfig.get_paths())"

# Verify Rust toolchain
rustc --version
cargo --version

# Clean and rebuild
cargo clean
rm -rf target/
maturin develop
```

## Advanced Topics

### 1. Custom Build Scripts

Create `build.rs` for custom compilation steps:

```rust
fn main() {
    // Custom build logic
    println!("cargo:rerun-if-changed=build.rs");

    // Example: Generate code
    // codegen::generate_bindings();
}
```

### 2. Feature Flags

Configure features in `Cargo.toml`:

```toml
[features]
default = ["cpu"]
cpu = []
simd = ["cpu", "scirs2-core/simd"]
gpu = ["scirs2-core/gpu"]
```

Build with features:
```bash
maturin build --release --features simd
```

### 3. Optimization Profiles

Add to `Cargo.toml`:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false
```

Build with profile:
```bash
cargo build --profile release-with-debug
```

### 4. Binary Size Optimization

```bash
# Install cargo-bloat
cargo install cargo-bloat

# Analyze binary size
cargo bloat --release -n 20

# Build with size optimization
RUSTFLAGS="-C opt-level=z" maturin build --release
```

### 5. Caching for Faster Builds

```bash
# Install sccache
cargo install sccache

# Configure Rust to use sccache
export RUSTC_WRAPPER=sccache

# Build (will cache compiled dependencies)
maturin build --release

# Check cache stats
sccache --show-stats
```

### 6. Multi-Package Workspace

For projects with multiple Python packages:

```toml
# workspace Cargo.toml
[workspace]
members = [
    "crates/pytensorlogic",
    "crates/pytensorlogic-gpu",
]
```

Build specific package:
```bash
maturin build -m crates/pytensorlogic/Cargo.toml
```

### 7. Conditional Compilation

```rust
#[cfg(feature = "simd")]
use simd_module;

#[pyfunction]
fn process_data(arr: PyArray1<f64>) -> f64 {
    #[cfg(feature = "simd")]
    {
        simd_process(arr)
    }

    #[cfg(not(feature = "simd"))]
    {
        scalar_process(arr)
    }
}
```

## Resources

### Documentation

- [Maturin Guide](https://maturin.rs/)
- [PyO3 User Guide](https://pyo3.rs/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Rust Book](https://doc.rust-lang.org/book/)

### Tools

- [Maturin GitHub](https://github.com/PyO3/maturin)
- [PyO3 GitHub](https://github.com/PyO3/pyo3)
- [cibuildwheel](https://cibuildwheel.readthedocs.io/) - For complex multi-platform builds

### Community

- [PyO3 Discord](https://discord.gg/PyO3)
- [Rust Users Forum](https://users.rust-lang.org/)
- [Python Packaging Forum](https://discuss.python.org/c/packaging/14)

## Quick Reference

### Essential Commands

```bash
# Development
maturin develop                    # Fast debug build
maturin develop --release          # Optimized build

# Building wheels
maturin build --release           # Build wheel
maturin build --release --features simd  # With features

# Testing
pytest tests/                     # Run Python tests
cargo test -p pytensorlogic      # Run Rust tests

# Publishing
maturin publish --no-sdist --repository testpypi  # Upload to TestPyPI
maturin publish --no-sdist                        # Upload to PyPI (recommended)

# Cleaning
cargo clean                       # Clean build artifacts
rm -rf target/wheels/*           # Clean wheels
```

### Environment Variables

```bash
# Rust compilation
export RUSTFLAGS="-C target-cpu=native"  # Optimize for CPU
export CARGO_INCREMENTAL=1               # Incremental compilation

# Maturin
export MATURIN_PEP517_ARGS="--features simd"  # Default features

# Python
export PYTHONPATH="$PWD/crates/pytensorlogic/python"  # Add to path
```

## Conclusion

This guide covers the complete packaging workflow for TensorLogic Python bindings. For questions or issues:

- Check [Troubleshooting](#troubleshooting) section
- Open an issue on [GitHub](https://github.com/cool-japan/tensorlogic/issues)
- Consult [Maturin documentation](https://maturin.rs/)

Happy packaging! 🎉
