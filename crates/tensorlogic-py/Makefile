# Makefile for TensorLogic Python Bindings
#
# Common tasks for developing, building, and distributing Python bindings

.PHONY: help dev install build build-release test test-python test-rust clean lint format check-format publish-test publish wheel wheels dev-simd

# Default target
help:
	@echo "TensorLogic Python Bindings - Make Targets"
	@echo ""
	@echo "Development:"
	@echo "  make dev            - Build and install in development mode (debug)"
	@echo "  make dev-simd       - Build with SIMD features (development)"
	@echo "  make install        - Build and install in release mode"
	@echo ""
	@echo "Building:"
	@echo "  make build          - Build wheel (debug)"
	@echo "  make build-release  - Build optimized wheel (release)"
	@echo "  make wheels         - Build wheels for all Python versions"
	@echo "  make wheel          - Alias for build-release"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests (Python + Rust)"
	@echo "  make test-python    - Run Python tests only"
	@echo "  make test-rust      - Run Rust tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linters (clippy + ruff)"
	@echo "  make format         - Format code (rustfmt + black)"
	@echo "  make check-format   - Check code formatting"
	@echo ""
	@echo "Publishing:"
	@echo "  make publish-test   - Publish to TestPyPI"
	@echo "  make publish        - Publish to PyPI (use with caution!)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove build artifacts"

# Development builds
dev:
	maturin develop

dev-simd:
	maturin develop --features simd

install:
	maturin develop --release

# Building wheels
build:
	maturin build --out target/wheels

build-release:
	maturin build --release --out target/wheels

wheel: build-release

# Build for multiple Python versions
wheels:
	maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12 --out target/wheels

# Testing
test: test-rust test-python

test-python:
	pytest tests/ -v

test-rust:
	cargo test -p tensorlogic-py

# Code quality
lint:
	cargo clippy --all-targets -- -D warnings
	ruff check . || true

format:
	cargo fmt --all
	black tests/ python_examples/ || true
	ruff format . || true

check-format:
	cargo fmt --all -- --check
	black --check tests/ python_examples/ || true

# Publishing
publish-test: clean build-release
	@echo "Publishing to TestPyPI..."
	maturin upload --repository testpypi target/wheels/*

publish: clean build-release
	@echo "⚠️  Publishing to PyPI (production)!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		maturin upload target/wheels/*; \
	else \
		echo "Cancelled."; \
	fi

# Cleanup
clean:
	cargo clean
	rm -rf target/wheels
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
