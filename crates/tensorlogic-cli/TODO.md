# TensorLogic CLI — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Command-line interface for TensorLogic: compile, explain, infer, train, REPL.

## Completed in v0.1.0

- Command-line argument parsing with clap (input/output formats, strategy selection, domain definitions, validation, debug flags).
- Expression parser supporting predicates, logical operators (AND/OR/NOT/IMPLIES), quantifiers (EXISTS/FORALL), arithmetic, comparisons, conditionals, and Unicode operators.
- Multiple input formats: expression strings, JSON, YAML, and stdin.
- Multiple output formats: human-readable graph, Graphviz DOT, JSON, and statistics.
- Six compilation strategy presets: soft_differentiable, hard_boolean, fuzzy_godel, fuzzy_product, fuzzy_lukasiewicz, probabilistic.
- Domain management via CLI (`--domain Name:size`) and configuration file with automatic inference.
- Graph validation with free-variable, arity, type, and structure checks, plus exit-code reporting.
- Comprehensive error handling with location tracking, suggestions, and typed exit codes (0–4).
- Debug mode with structured output for parsed expressions, context, intermediate steps, and results.
- Interactive REPL with multi-line editing, persistent history, session state, and dot-commands (.help, .domain, .strategy, .execute, .optimize, .profile, .cache, .macro, etc.).
- Configuration file support (`.tensorlogicrc`) with user/project/environment resolution and init/edit commands.
- Batch processing with parallel (rayon) compilation, progress bars, and cache warming from pipe-delimited files.
- Watch mode with `notify`-based file watching, configurable debounce, and live reload.
- Graph analysis with complexity metrics, FLOPs/memory estimation, and operation breakdown (`--analyze`).
- Format conversion between expression/JSON/YAML with pretty-printing (`convert --from --to`).
- Execute command with multiple backends (cpu, parallel, profiled) and table/JSON/CSV/NumPy output.
- Optimize command with real passes (identity elimination, einsum merging, contraction ordering) at four levels.
- Benchmark and profile commands with phase timing, memory estimation, statistical analysis, and JSON export.
- Persistent compilation cache with LRU eviction, gzip compression, analytics/efficiency scoring, and `cache stats|clear|path` commands.
- Expression simplification: double-negation, idempotent/absorption/De Morgan laws, constant folding, identity/annihilation.
- Visualization: `DotExporter` (clustered Graphviz) and `AsciiRenderer` (terminal tree view).
- Shell completion generation for bash, zsh, fish, and PowerShell via `clap_complete`.
- Library mode (`lib.rs`) with public API, type aliases, and example programs for embedding.
- Macro system with parameterized definitions, recursive expansion, built-ins (transitive/symmetric/reflexive/antisymmetric/total), and REPL integration.
- FFI bindings for C/C++ (`tensorlogic.h`) and Python (`tensorlogic_ffi.py`) covering compilation, execution, optimization, and benchmarking.
- Documentation: README, 30-recipe cookbook, Unix man page, 5 real-world `.tl` examples, and CI/CD examples (GitHub Actions, GitLab CI, Jenkins, Docker).
- Test suite: 291 passing tests (unit, integration, end-to-end, executor, macro, FFI, cache, simplification, library, batch, analytics, snapshot).

## v0.1.x Stabilization (in-repo, unblocked)

- [x] ~~**Rename the CLI `[[bin]]` from `tensorlogic` to `tensorlogic-cli`** in `crates/tensorlogic-cli/Cargo.toml` to fix the doc-build bin/lib output-path collision warning (Cargo issue #6313).~~ (completed 2026-04-15)
- Interactive Mode: add Ctrl+R history search in the REPL (rustyline reverse-incremental search).
- Config: add TOML-based default config loader discovery for additional well-known locations.
- Incremental compilation for the persistent cache (currently marked FUTURE in the cache module).
- Lazy / on-demand module loading to reduce startup time for large sessions.
- Re-enable the commented-out `tlc` short-name binary alias once the primary rename lands.
- Expand tutorial/video walkthroughs referenced in the documentation roadmap.

## v0.2.0 Roadmap

- LSP (Language Server Protocol) for IDE integration (VS Code, Neovim, etc.).
- REST API server mode for HTTP-based compilation and JSON API access.
- TUI (Text User Interface) using ratatui for an interactive compile/inspect experience.
- Shell completion generation improvements (dynamic argument completion for domains/strategies) and additional shells.
- Remote execution backend (SSH / gRPC) for offloading graph execution to remote workers.
- Plugin system for custom input formats, output formats, and strategies.
- Web-based UI with a browser compilation front-end, visual graph editor, and interactive debugger.

## Design Notes

### Source layout (~8,900 lines implementation, ~5,300 lines docs/examples)

```
analysis.rs           ~227 lines  - Graph metrics and complexity analysis
batch.rs              ~299 lines  - Parallel batch processing
benchmark.rs          ~337 lines  - Performance benchmarking
cache.rs             ~1042 lines  - LRU cache with analytics & warmup
cli.rs                ~345 lines  - Clap CLI definitions
completion.rs          ~24 lines  - Shell completion generation
config.rs             ~251 lines  - Configuration file support
conversion.rs         ~394 lines  - Format conversion and pretty-printing
executor.rs           ~456 lines  - Execution engine with backend selection
ffi.rs                ~704 lines  - FFI bindings for C/C++ integration
lib.rs                ~161 lines  - Library API and public exports
macros.rs             ~554 lines  - Macro system with expansion engine
main.rs               ~725 lines  - Main entry point and command routing
optimize.rs           ~296 lines  - Optimization pipeline
output.rs              ~44 lines  - Colored output formatting
parser.rs             ~393 lines  - Enhanced expression parser
profile.rs           ~1071 lines  - Profiling with execution metrics
repl.rs               ~590 lines  - Interactive REPL mode
simplify.rs           ~669 lines  - Expression simplification
visualization.rs      ~200 lines  - DotExporter and AsciiRenderer
watch.rs              ~113 lines  - File watching and auto-recompilation
tests/cli_integration ~400 lines  - Integration tests (32 tests)
tests/end_to_end      ~410 lines  - End-to-end tests (20 tests)
tests/executor_integ   ~80 lines  - Executor integration tests
```

### Binary names

- `tensorlogic` (primary; slated to rename to `tensorlogic-cli` — see v0.1.x Stabilization).
- `tlc` (short alias, currently commented out for backward compatibility).

### Key dependencies

- `clap` 4.5 and `clap_complete` 4.5 — argument parsing and shell completion.
- `rustyline` 14.0 — REPL with history.
- `colored` 2.1, `indicatif` 0.17 — terminal UX.
- `notify` 6.1 — filesystem watching for watch mode.
- `dirs` 5.0, `toml` 0.8, `chrono` 0.4 — config discovery and timestamps.

### Notes

- CLI is feature-complete for the v0.1.0 stable release.
- Build status: zero errors, zero warnings.
- All dependencies use `workspace = true` per the Workspace Policy.
