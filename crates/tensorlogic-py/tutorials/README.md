# TensorLogic Python Tutorials

This directory contains comprehensive Jupyter notebooks demonstrating the TensorLogic Python API.

## Getting Started

### Prerequisites

```bash
# Install TensorLogic Python bindings
cd crates/pytensorlogic
maturin develop

# Install tutorial dependencies
pip install jupyter matplotlib numpy
```

### Launching Jupyter

```bash
cd crates/pytensorlogic/tutorials
jupyter notebook
```

## Tutorials

### 01_getting_started.ipynb

**Level:** Beginner
**Duration:** 45-60 minutes
**Topics:**
- Basic logical expressions (predicates, AND, OR, NOT)
- Compilation and execution workflow
- Compilation strategies (soft, hard, fuzzy, probabilistic)
- Quantifiers (EXISTS, FORALL)
- Arithmetic operations (add, sub, mul, div)
- Comparison operations (eq, lt, gt, lte, gte)
- Conditional expressions (IF-THEN-ELSE)
- Complex nested expressions (De Morgan's laws, implication)
- Adapter types (DomainInfo, PredicateInfo, SymbolTable, CompilerContext)
- Practical example: Social network reasoning

**Learning Outcomes:**
- Understand the TensorLogic compilation model
- Create and execute logical expressions
- Choose appropriate compilation strategies
- Visualize results with matplotlib

### 02_advanced_topics.ipynb

**Level:** Advanced
**Duration:** 60-90 minutes
**Topics:**
- Multi-arity predicates (binary, ternary, n-ary)
- Relational reasoning (transitive closure, path queries)
- Nested quantifiers (double, triple quantification)
- Performance optimization and benchmarking
- Graph inspection and analysis
- Strategy selection guide with use cases
- Integration patterns (iterative reasoning, multi-rule systems)
- Error handling and debugging techniques
- Best practices and performance tips
- Type safety patterns

**Learning Outcomes:**
- Master complex relational queries
- Optimize performance for production use
- Build robust integration pipelines
- Debug compilation and execution issues
- Apply best practices for maintainable code

## Tutorial Structure

Each notebook follows this structure:

1. **Setup** - Import dependencies and verify installation
2. **Conceptual Introduction** - Explain the topic with examples
3. **Code Examples** - Executable code demonstrating features
4. **Visualizations** - Plots and diagrams for understanding
5. **Practice** - Exercises and challenges (where applicable)
6. **Summary** - Key takeaways and next steps

## Running the Tutorials

### Option 1: Interactive (Recommended)

```bash
jupyter notebook
```

Open the desired notebook and execute cells interactively.

### Option 2: Non-Interactive

```bash
jupyter nbconvert --to notebook --execute 01_getting_started.ipynb
```

This executes the notebook and saves the output.

### Option 3: Python Script

```bash
jupyter nbconvert --to python 01_getting_started.ipynb
python 01_getting_started.py
```

Converts the notebook to a Python script for non-interactive execution.

## Tips for Learning

1. **Execute cells in order** - Notebooks have state, cells depend on previous execution
2. **Experiment** - Modify examples to test understanding
3. **Visualize** - Use matplotlib to understand tensor operations
4. **Debug** - Use the debug helpers provided in advanced tutorials
5. **Read tests** - The `tests/` directory has additional examples
6. **Ask questions** - File issues on GitHub for clarification

## Common Issues

### Issue: ModuleNotFoundError: No module named 'pytensorlogic'

**Solution:**
```bash
cd crates/pytensorlogic
maturin develop
```

### Issue: Compilation fails with Rust errors

**Solution:** Ensure you have Rust toolchain installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Issue: Jupyter kernel crashes

**Solution:** Install kernel dependencies:
```bash
pip install ipykernel
python -m ipykernel install --user
```

### Issue: Matplotlib plots not showing

**Solution:** For notebook:
```python
%matplotlib inline
```

For JupyterLab:
```bash
pip install jupyterlab
```

## Additional Resources

- **API Reference**: See `pytensorlogic.pyi` for complete type signatures
- **Test Suite**: `tests/` directory for comprehensive usage examples
- **Examples**: `examples/` directory for standalone demonstrations
- **Documentation**: `README.md` and `CLAUDE.md` in repository root
- **Paper**: TensorLogic research paper (link in main README)

## Contributing

Found an error or have a suggestion? Please:
1. Check existing issues on GitHub
2. File a new issue with details
3. Submit a pull request with fixes/improvements

We welcome contributions to improve these tutorials!

## License

These tutorials are part of the TensorLogic project and are licensed under Apache-2.0.

---

**Last Updated:** 2025-11-04
**Maintainer:** COOLJAPAN ecosystem
**Feedback:** https://github.com/cool-japan/tensorlogic/issues
