# bench-tracker

A comprehensive benchmark regression tracking tool for TensorLogic.

## Features

- **Save Baselines**: Capture current benchmark results as a baseline for future comparisons
- **Compare Results**: Compare current benchmarks against saved baselines with configurable thresholds
- **Detect Regressions**: Automatically identify performance regressions and improvements
- **Multiple Report Formats**: Generate reports in text, JSON, or HTML format
- **Detailed Statistics**: View comprehensive statistics for individual benchmarks
- **Git Integration**: Automatically tracks git commit hashes with baselines

## Installation

The tool is part of the TensorLogic workspace. Build it with:

```bash
cargo build -p bench-tracker
```

## Usage

### Save Current Benchmarks as Baseline

```bash
cargo run -p bench-tracker -- save --name "my-baseline" --output benchmarks/baseline.json
```

Options:
- `--criterion-dir`: Path to criterion output directory (default: `target/criterion`)
- `--output`: Output path for baseline file (default: `benchmarks/baseline.json`)
- `--name`: Baseline name/tag (optional)

### List Saved Baselines

```bash
cargo run -p bench-tracker -- list --baseline benchmarks/baseline.json
```

### Compare Against Baseline

```bash
cargo run -p bench-tracker -- compare --threshold 5.0
```

Options:
- `--criterion-dir`: Path to criterion output directory (default: `target/criterion`)
- `--baseline`: Path to baseline file (default: `benchmarks/baseline.json`)
- `--threshold`: Regression threshold in percentage (default: `5.0`)
- `--format`: Output format: `text`, `json`, or `html` (default: `text`)

The compare command will:
- Show a detailed comparison table
- Highlight regressions in red
- Highlight improvements in green
- Mark stable benchmarks in blue
- Exit with error code if regressions are detected

### View Detailed Statistics

```bash
cargo run -p bench-tracker -- stats --name e2e_simple_predicate
```

Shows detailed statistical information including:
- Mean with confidence intervals
- Median with confidence intervals
- Standard deviation with confidence intervals
- All measurements for each parameter variant

## Example Workflow

```bash
# 1. Run benchmarks and save baseline
cargo bench --bench end_to_end
cargo run -p bench-tracker -- save --name "v0.1.0-baseline"

# 2. Make code changes
# ... edit code ...

# 3. Run benchmarks again
cargo bench --bench end_to_end

# 4. Compare against baseline
cargo run -p bench-tracker -- compare --threshold 5.0

# 5. View detailed stats for specific benchmark
cargo run -p bench-tracker -- stats --name e2e_training
```

## CI Integration

The tool is designed to be used in CI pipelines:

```bash
# In CI script
cargo bench --bench end_to_end
cargo run -p bench-tracker -- compare --threshold 5.0 --format json > regression-report.json

# Exit code is non-zero if regressions detected
```

## Output Format

### Text Report

Default format with colored output and formatted table showing:
- Benchmark name
- Baseline and current measurements
- Percentage change
- Status (REGRESSION/IMPROVEMENT/STABLE)
- Summary statistics

### JSON Report

Machine-readable format suitable for CI/CD integration:

```json
{
  "baseline": {
    "name": "baseline-name",
    "created_at": "2025-12-31T09:46:58.747394Z",
    "commit": "23955b3..."
  },
  "comparisons": [
    {
      "name": "e2e_simple_predicate",
      "parameter": "10",
      "baseline_mean_ns": 113.48,
      "current_mean_ns": 113.48,
      "change_percent": 0.0,
      "is_regression": false,
      "is_improvement": false
    }
  ]
}
```

### HTML Report

Formatted HTML report with:
- Color-coded status indicators
- Sortable tables
- Summary statistics
- Responsive design

## Baseline File Format

Baselines are stored as JSON with complete statistical information:

```json
{
  "name": "baseline-name",
  "created_at": "2025-12-31T09:46:58.747394Z",
  "commit": "23955b3caa7a14b2825572e9b032fff218411238",
  "results": {
    "benchmark_name/parameter": {
      "name": "benchmark_name",
      "parameter": "parameter",
      "estimates": {
        "mean": {
          "point_estimate": 113.48,
          "standard_error": 1.47,
          "confidence_interval": {
            "confidence_level": 0.95,
            "lower_bound": 110.97,
            "upper_bound": 116.70
          }
        },
        ...
      },
      "timestamp": "2025-12-31T09:47:23.126650Z"
    }
  }
}
```

## License

Licensed under Apache-2.0.

## Authors

COOLJAPAN OU (Team Kitasan)
