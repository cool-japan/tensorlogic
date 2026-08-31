# TensorLogic Python Bindings — TODO

**Status**: Alpha | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

PyO3 / abi3-py39 Python bindings via maturin.

## Completed

### Infrastructure
- [x] Basic PyO3 structure
- [x] abi3-py39 configuration
- [x] NumPy integration
- [x] Module organization (types, compiler, executor, numpy_conversion)
- [x] Maturin build with zero warnings

### Core Types Binding - COMPLETE
- [x] **PyTerm** - Expose Term to Python with __repr__ and __str__
- [x] **PyTLExpr** - Expose TLExpr with all logical operations
- [x] **PyEinsumGraph** - Expose compiled graphs with stats
- [x] Helper functions: var(), const(), pred(), and_(), or_(), not_(), exists(), forall(), imply(), constant()

### Compilation API - COMPLETE
- [x] **py_compile()** - Main compilation function
- [x] **PyCompilationConfig** - Configuration wrapper with 6 presets
  - [x] soft_differentiable() - Default for neural training
  - [x] hard_boolean() - Discrete Boolean logic
  - [x] fuzzy_godel() - Godel fuzzy logic
  - [x] fuzzy_product() - Product fuzzy logic
  - [x] fuzzy_lukasiewicz() - Lukasiewicz fuzzy logic
  - [x] probabilistic() - Probabilistic interpretation
- [x] **py_compile_with_config()** - Compilation with custom config
- [x] **py_compile_with_context()** - Compilation with CompilerContext
- [x] Error handling with PyRuntimeError

### Execution API - COMPLETE
- [x] **py_execute()** - Execute graphs with NumPy inputs
- [x] Dynamic tensor shape handling (ArrayD<f64>)
- [x] Input/output via Python dictionaries
- [x] Integration with Scirs2Exec backend
- [x] Proper error propagation to Python

### NumPy Integration - COMPLETE
- [x] **NumPy interop module** (numpy_conversion.rs)
  - [x] numpy_to_array2() - Convert 2D arrays
  - [x] array2_to_numpy() - Export 2D arrays
  - [x] numpy_to_arrayd() - Convert dynamic arrays
  - [x] arrayd_to_numpy() - Export dynamic arrays
- [x] Proper lifetime management with PyReadonlyArray
- [x] Safe memory handling with readwrite() slices
- [ ] Zero-copy optimization (requires unsafe improvements) - FUTURE
- [ ] **PyTorch interop** - FUTURE

### Python-Friendly API - COMPLETE
- [x] Pythonic naming (snake_case for all functions)
- [x] Comprehensive docstrings with Args/Returns/Example
- [x] __repr__ implementations for all types
- [x] __str__ implementations using pretty-printing
- [x] Proper error messages
- [x] **Type hints (.pyi stub files)** (1100+ lines)
- [x] **Context managers**
  - [x] ExecutionContext - Managed graph execution
  - [x] CompilationContext - Managed compilation

### Arithmetic Operations - COMPLETE
- [x] **add()** - Addition operation (left + right)
- [x] **sub()** - Subtraction operation (left - right)
- [x] **mul()** - Multiplication operation (left * right)
- [x] **div()** - Division operation (left / right)
- [x] Full integration with compilation and execution
- [x] Comprehensive examples (examples/arithmetic_operations.py)
- [x] Full test coverage (tests/test_types.py, tests/test_execution.py)

### Comparison Operations - COMPLETE
- [x] **eq()** - Equality comparison (left == right)
- [x] **lt()** - Less than comparison (left < right)
- [x] **gt()** - Greater than comparison (left > right)
- [x] **lte()** - Less than or equal (left <= right)
- [x] **gte()** - Greater than or equal (left >= right)
- [x] Full integration with compilation and execution
- [x] Comprehensive examples (examples/comparison_conditionals.py)
- [x] Full test coverage (tests/test_types.py, tests/test_execution.py)

### Conditional Operations - COMPLETE
- [x] **if_then_else()** - Conditional expression (ternary operator)
- [x] Support for nested conditionals
- [x] Comprehensive examples (examples/comparison_conditionals.py)
- [x] Full test coverage (tests/test_types.py, tests/test_execution.py)

### Development Infrastructure - COMPLETE
- [x] **pytensorlogic.pyi** - Complete type stubs for IDE support (1100+ lines)
- [x] **pytest test suite** - 300+ tests covering all operations
  - [x] test_types.py - Type creation and operation tests
  - [x] test_execution.py - End-to-end execution tests
  - [x] test_backend.py - Backend selection tests
  - [x] test_provenance.py - Provenance tracking tests (40+ tests)
  - [x] test_training.py - Training API tests (40+ tests)
  - [x] test_persistence.py - Model persistence tests (20+ tests)
  - [x] test_dsl.py - Rule Builder DSL tests (100+ tests)
- [x] **pytest.ini** - Test configuration
- [x] **requirements-dev.txt** - Development dependencies
- [x] **Python examples** - 12 runnable demonstration scripts
  - [x] arithmetic_operations.py - All arithmetic operations
  - [x] comparison_conditionals.py - All comparisons and conditionals
  - [x] basic_usage.py - Comprehensive usage guide
  - [x] backend_selection.py - Backend selection
  - [x] provenance_tracking.py - Provenance tracking (450+ lines)
  - [x] training_workflow.py - Training API (450+ lines, 10 scenarios)
  - [x] model_persistence.py - Model persistence (600+ lines, 10 scenarios)
  - [x] rule_builder_dsl.py - Rule Builder DSL (550+ lines, 10 examples)
  - [x] async_execution_demo.py - Async execution (300+ lines)
  - [x] performance_benchmark.py - Performance benchmarks
  - [x] memory_profiling.py - Memory profiling and streaming
  - [x] advanced_symbol_table.py - SymbolTable and CompilerContext

### Advanced Domain Management - COMPLETE
- [x] **DomainInfo** - Domain representation with metadata
  - [x] name, cardinality properties
  - [x] description, elements support
  - [x] set_description(), set_elements() methods
- [x] **PredicateInfo** - Predicate representation
  - [x] name, arity, arg_domains properties
  - [x] description support
- [x] **SymbolTable** - Complete symbol table management
  - [x] add_domain(), add_predicate()
  - [x] bind_variable(), get_domain(), get_predicate()
  - [x] get_variable_domain(), list_domains(), list_predicates()
  - [x] infer_from_expr() - Automatic schema inference
  - [x] get_variable_bindings() - Query all bindings
  - [x] to_json() / from_json() - JSON serialization
- [x] **CompilerContext** - Low-level compilation control
  - [x] add_domain(), bind_var()
  - [x] assign_axis() - Einsum axis assignment
  - [x] fresh_temp() - Temporary tensor names
  - [x] get_domains(), get_variable_bindings(), get_axis_assignments()
  - [x] get_variable_domain(), get_variable_axis()
- [x] **Comprehensive example** (examples/advanced_symbol_table.py)

### Backend Selection API - COMPLETE
- [x] **PyBackend** - Backend enumeration (AUTO, SCIRS2_CPU, SCIRS2_SIMD, SCIRS2_GPU)
- [x] **PyBackendCapabilities** - Backend capability information
  - [x] name, version, devices, dtypes, features, max_dims properties
  - [x] supports_device(), supports_dtype(), supports_feature() methods
  - [x] summary(), to_dict() methods
- [x] get_backend_capabilities() - Query backend capabilities
- [x] list_available_backends() - List all backends with availability
- [x] get_default_backend() - Get default backend
- [x] get_system_info() - System and backend information
- [x] Backend parameter in py_execute()

### Provenance Tracking - COMPLETE
- [x] **SourceLocation** - Source code location tracking
  - [x] file, line, column properties
  - [x] String representation
- [x] **SourceSpan** - Source code span representation
  - [x] start, end locations
  - [x] Span formatting
- [x] **Provenance** - Provenance metadata for IR nodes
  - [x] rule_id, source_file, span properties
  - [x] Custom attributes (add_attribute, get_attribute, get_attributes)
  - [x] Full Python bindings
- [x] **ProvenanceTracker** - RDF and tensor computation mappings
  - [x] track_entity() - Entity to tensor mappings
  - [x] track_shape() - SHACL shape to rule mappings
  - [x] track_inferred_triple() - RDF* triple tracking
  - [x] get_entity(), get_tensor() - Bidirectional lookups
  - [x] get_entity_mappings(), get_shape_mappings()
  - [x] get_high_confidence_inferences() - Confidence filtering
  - [x] to_rdf_star(), to_rdfstar_turtle() - RDF* export
  - [x] to_json(), from_json() - JSON serialization
  - [x] RDF* support with enable_rdfstar flag
- [x] **Graph Provenance Functions**
  - [x] get_provenance() - Extract provenance from graphs
  - [x] get_metadata() - Extract metadata from graphs
  - [x] provenance_tracker() - Helper function
- [x] **Type stubs** - pytensorlogic.pyi updated
- [x] **Test suite** - test_provenance.py (300+ lines, 40+ tests)
- [x] **Example** - provenance_tracking.py (450+ lines, 10 scenarios)

### Training API - COMPLETE
- [x] **Loss Functions** - Multiple loss function implementations
  - [x] mse_loss() - Mean Squared Error for regression
  - [x] bce_loss() - Binary Cross-Entropy for binary classification
  - [x] cross_entropy_loss() - Cross-Entropy for multi-class classification
  - [x] LossFunction class with __call__ method
- [x] **Optimizers** - Optimizer implementations for parameter updates
  - [x] sgd() - Stochastic Gradient Descent with momentum
  - [x] adam() - Adam optimizer with beta1, beta2, epsilon
  - [x] rmsprop() - RMSprop optimizer with alpha, epsilon
  - [x] Learning rate adjustment support
- [x] **Callbacks** - Training monitoring and control
  - [x] early_stopping() - Early stopping with patience and min_delta
  - [x] model_checkpoint() - Model checkpointing during training
  - [x] logger() - Training progress logging with verbosity control
- [x] **Trainer Class** - High-level training interface
  - [x] fit() method with epochs, validation_data, verbose
  - [x] evaluate() method for model evaluation
  - [x] predict() method for inference
  - [x] TrainingHistory tracking with metrics
- [x] **Convenience Functions**
  - [x] fit() function for quick training without explicit Trainer
- [x] **Type stubs** - pytensorlogic.pyi updated with training types
- [x] **Test suite** - test_training.py (370+ lines, 40+ tests)
- [x] **Example** - training_workflow.py (450+ lines, 10 scenarios)
- [x] **Code quality** - Zero clippy warnings, SCIRS2 compliant

### Model Persistence - COMPLETE
- [x] **ModelPackage** - Complete model serialization container
  - [x] graph, config, symbol_table, parameters, metadata properties
  - [x] add_metadata(), get_metadata() - Metadata management
  - [x] save_json(), load_json() - JSON format (human-readable)
  - [x] save_binary(), load_binary() - Binary format (compact)
  - [x] to_json(), from_json() - JSON string conversion
  - [x] to_bytes(), from_bytes() - Binary conversion
  - [x] __getstate__, __setstate__ - Pickle support
- [x] **Persistence Functions**
  - [x] save_model() - Save compiled graphs
  - [x] load_model() - Load compiled graphs
  - [x] save_full_model() - Save with config and metadata
  - [x] load_full_model() - Load complete models
  - [x] model_package() - Helper function
- [x] **Format Support**
  - [x] JSON format (human-readable, cross-platform)
  - [x] Binary format (compact, efficient)
  - [x] Auto format detection from file extension
  - [x] Pickle support for Python workflows
- [x] **Type stubs** - pytensorlogic.pyi updated (200+ lines)
- [x] **Test suite** - test_persistence.py (400+ lines, 20+ tests)
- [x] **Example** - model_persistence.py (600+ lines, 10 scenarios)
- [ ] ONNX export - FUTURE

### Rule Builder DSL - COMPLETE
- [x] Var class with domain bindings
- [x] PredicateBuilder for callable predicates with arity/domain validation
- [x] Operator overloading (&, |, ~, >>)
- [x] RuleBuilder context manager
- [x] Symbol table integration
- [x] Multiple compilation strategies
- [x] Comprehensive examples and tests
- [x] Full type stubs

### Jupyter Integration - COMPLETE
- [x] **Rich HTML Display** - `_repr_html_()` methods for all major types
  - [x] EinsumGraph - Node statistics and type breakdown
  - [x] SymbolTable - Domains, predicates, variables in tables
  - [x] CompilationConfig - Configuration semantics
  - [x] ModelPackage - Component checklist and metadata
  - [x] TrainingHistory - Epoch-by-epoch loss tables
  - [x] Provenance - Rule origin and attributes
- [x] **HTML Generation Module** (jupyter.rs)
  - [x] HTML table generator
  - [x] Card/badge components
  - [x] Key-value list formatter
  - [x] Specialized visualizers for each type
- [ ] Visualization widgets - FUTURE
- [ ] Interactive debugging - FUTURE
- [ ] Progress bars - FUTURE

### Performance Monitoring - COMPLETE
- [x] **GIL Release** - Release GIL during CPU-bound tensor operations
- [x] **Parallel execution** - BatchExecutor and execute_parallel
- [x] **Async support** - AsyncResult, execute_async
- [x] **Memory profiling** - Complete performance monitoring
  - [x] MemorySnapshot class
  - [x] Profiler class with timing statistics
  - [x] Timer context manager
  - [x] memory_snapshot(), get_memory_info(), reset_memory_tracking()

### Streaming Execution - COMPLETE
- [x] **StreamingExecutor** - Process large datasets in chunks
  - [x] Configurable chunk_size and overlap
  - [x] execute_streaming() method
- [x] **DataGenerator** - Memory-efficient data loading
- [x] **ResultAccumulator** - Accumulate streaming results
  - [x] add(), combine(), stats()
- [x] **process_stream()** - Process iterator through graph

### Async Cancellation - COMPLETE
- [x] **CancellationToken** - Cancel async operations
  - [x] cancel(), is_cancelled(), reset()
- [x] **AsyncResult cancellation support**
  - [x] cancel(), is_cancelled(), get_cancellation_token()

### Utility Functions & Context Managers - COMPLETE
- [x] **Custom Exceptions** - Better error handling
  - [x] CompilationError - Compilation failures
  - [x] ExecutionError - Execution failures
  - [x] ValidationError - Input validation failures
  - [x] BackendError - Backend operations failures
  - [x] ConfigurationError - Invalid configuration
- [x] **ExecutionContext** - Context manager for execution
  - [x] execute(), get_results(), execution_count(), clear_results()
- [x] **CompilationContext** - Context manager for compilation
  - [x] compile(), get_graphs(), get_graph(), graph_count()
- [x] **Utility Functions**
  - [x] quick_execute() - One-liner compile + execute
  - [x] validate_inputs() - Input validation
  - [x] batch_compile() - Compile multiple expressions
  - [x] batch_predict() - Predict on multiple inputs

### Documentation - COMPLETE
- [x] **Comprehensive README.md**
- [x] **QUICKSTART.md** (Quick start guide)
- [x] **examples/README.md** (Example navigation)
- [x] **Complete API reference** (in README.md)
- [x] **COMPLIANCE_REPORT.md** - Quality validation report
- [x] **PACKAGING.md** - Complete packaging guide
- [x] **pytensorlogic.pyi** - Complete with all types (1100+ lines)
- [ ] Sphinx documentation - FUTURE
- [ ] Tutorial Jupyter notebooks - FUTURE

## v0.1.3 Enhancements (2026-03-30)

- [x] **Training Progress Callbacks** (`progress.rs`): `PyProgressEvent` (step/loss/grad_norm/elapsed_ms with tqdm-compatible `as_dict()`), `PyCompilationEvent` (6 phase events), `PyTrainingResult` (final/avg/reduction loss), `PyTrainingLoop` with optional Python callback. 12 new tests.

## Future Enhancements

### Integrations
- [ ] PyTorch tensor integration
- [ ] GPU backend support
- [ ] ONNX export

### Packaging
- [ ] PyPI release
- [ ] maturin build and wheel distribution
- [ ] Platform-specific builds (Linux x86_64, macOS arm64/x86_64, Windows)

### Testing
- [ ] Coverage reporting in CI (pytest-cov configured, needs CI pipeline)
- [ ] Benchmark suite in CI (pytest-benchmark configured)

### Advanced Features
- [ ] Visualization widgets for Jupyter
- [ ] Interactive debugging
- [ ] Progress bars for long operations
- [ ] mypy strict mode validation

---

**Total Items:** 120+ tasks
**Completion:** 100% of core + medium + performance + utility features
**Release:** v0.1.0 Alpha (2026-04-06)

### Completion Summary
- Phase 1 Complete: Core types binding (PyTerm, PyTLExpr, PyEinsumGraph)
- Phase 2 Complete: Compilation API (compile, config presets)
- Phase 3 Complete: Execution API (execute with NumPy)
- Phase 4 Complete: NumPy interop (bidirectional conversion)
- Phase 5 Complete: Python-friendly API (docstrings, repr, error handling)
- Phase 6 Complete: Arithmetic operations (add, sub, mul, div)
- Phase 7 Complete: Comparison operations (eq, lt, gt, lte, gte)
- Phase 8 Complete: Conditional operations (if_then_else)
- Phase 9 Complete: Type stubs (.pyi) and testing infrastructure
- Phase 10 Complete: SymbolTable and domain management
- Phase 11 Complete: CompilerContext for advanced compilation
- Phase 12 Complete: Backend selection API (CPU/SIMD/GPU)
- Phase 13 Complete: Provenance tracking (full RDF* support)
- Phase 14 Complete: Training API (loss functions, optimizers, callbacks)
- Phase 15 Complete: Model Persistence (save/load, multiple formats, pickle)
- Phase 16 Complete: Jupyter Integration (rich HTML display for all types)
- Phase 17 Complete: Rule Builder DSL (Python-native syntax, operator overloading)
- Phase 18 Complete: Performance Monitoring (GIL release, profiler, memory tracking)
- Phase 19 Complete: Streaming Execution (StreamingExecutor, ResultAccumulator)
- Phase 20 Complete: Async Cancellation (CancellationToken, cancel support)
- Phase 21 Complete: Utility Functions (context managers, custom exceptions, helpers)

### Build Status
- Maturin build succeeds with zero warnings (run: `maturin develop`)
- Release build optimized and ready
- All dependencies resolved
- Zero clippy warnings
- Note: `cargo nextest` does not run Python integration tests; use `pytest tests/` after `maturin develop`

### Test & Example Status
- 300+ pytest tests across 7 test files (test_types, test_execution, test_backend, test_provenance, test_training, test_persistence, test_dsl)
- 12 comprehensive examples
- Type stub file (pytensorlogic.pyi) with full API coverage (1100+ lines)
- pytest.ini configuration
- requirements-dev.txt with all dependencies

### API Surface
- **Core:** var(), const(), pred(), and_(), or_(), not_(), exists(), forall(), imply(), constant(), if_then_else()
- **Arithmetic:** add(), sub(), mul(), div()
- **Comparisons:** eq(), lt(), gt(), lte(), gte()
- **Compilation:** compile(), compile_with_config(), compile_with_context()
- **Execution:** execute(), execute_async(), execute_parallel()
- **Async:** AsyncResult, BatchExecutor, CancellationToken, cancellation_token()
- **Adapters:** DomainInfo, PredicateInfo, SymbolTable, CompilerContext
- **Backend:** Backend, BackendCapabilities, get_backend_capabilities(), list_available_backends(), get_default_backend(), get_system_info()
- **Provenance:** SourceLocation, SourceSpan, Provenance, ProvenanceTracker, get_provenance(), get_metadata(), provenance_tracker()
- **Training:** LossFunction, Optimizer, Callback, TrainingHistory, Trainer, mse_loss(), bce_loss(), cross_entropy_loss(), sgd(), adam(), rmsprop(), early_stopping(), model_checkpoint(), logger(), fit()
- **Persistence:** ModelPackage, model_package(), save_model(), load_model(), save_full_model(), load_full_model()
- **DSL:** Var, PredicateBuilder, RuleBuilder, var_dsl(), pred_dsl(), rule_builder()
- **Performance:** MemorySnapshot, Profiler, Timer, memory_snapshot(), profiler(), timer(), get_memory_info()
- **Streaming:** StreamingExecutor, DataGenerator, ResultAccumulator, streaming_executor(), result_accumulator(), process_stream()
- **Exceptions:** CompilationError, ExecutionError, ValidationError, BackendError, ConfigurationError
- **Utils:** ExecutionContext, CompilationContext, quick_execute(), validate_inputs(), batch_compile(), batch_predict(), execution_context(), compilation_context()

**Total API:** 80+ functions, 35+ classes, 5 custom exceptions, 6 compilation strategies, 3 serialization formats, 6 rich displays, 4 operator overloads
