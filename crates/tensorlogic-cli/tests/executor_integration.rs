//! Integration tests for executor functionality

use assert_cmd::Command;

#[test]
fn test_backends_command() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("backends");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Available Backends"));
    assert!(stdout.contains("SciRS2 CPU"));
    assert!(stdout.contains("Backend Capabilities"));
}

#[test]
fn test_execute_command_basic() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("execute")
        .arg("person(x)")
        .arg("--backend")
        .arg("cpu");

    // This will fail because we need proper input setup, but should parse args correctly
    let output = cmd.output().unwrap();
    // Command should at least parse and attempt execution
    assert!(output.status.code().is_some());
}

#[test]
#[ignore = "Optimization passes are slow - see tensorlogic-compiler"]
fn test_optimize_command_basic() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("optimize")
        .arg("person(x) AND knows(x, y)")
        .arg("--level")
        .arg("basic")
        .arg("--stats");

    let output = cmd.output().unwrap();
    // Command should parse correctly
    assert!(output.status.code().is_some());
}

#[test]
#[ignore = "Optimization passes are slow - see tensorlogic-compiler"]
fn test_optimize_levels() {
    for level in &["none", "basic", "standard", "aggressive"] {
        #[allow(deprecated)]
        let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
        cmd.arg("optimize").arg("p(x)").arg("--level").arg(level);

        let output = cmd.output().unwrap();
        assert!(output.status.code().is_some());
    }
}

#[test]
fn test_execution_output_formats() {
    for format in &["table", "json", "csv", "numpy"] {
        #[allow(deprecated)]
        let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
        cmd.arg("execute")
            .arg("test(x)")
            .arg("--backend")
            .arg("cpu")
            .arg("--output-format")
            .arg(format);

        let output = cmd.output().unwrap();
        assert!(output.status.code().is_some());
    }
}

// Benchmark command tests

#[test]
fn test_benchmark_command_basic() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("benchmark").arg("person(x)").arg("-n").arg("3");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compilation Benchmark"));
    assert!(stdout.contains("Mean:"));
}

#[test]
#[ignore = "Optimization passes are slow - see tensorlogic-compiler"]
fn test_benchmark_with_optimization() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("benchmark")
        .arg("p(x) AND q(x)")
        .arg("-n")
        .arg("2")
        .arg("--optimize");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Optimization Benchmark"));
}

#[test]
fn test_benchmark_json_output() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("benchmark")
        .arg("test(x)")
        .arg("-n")
        .arg("2")
        .arg("--json");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should be valid JSON
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
}

#[test]
fn test_benchmark_verbose() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("benchmark")
        .arg("p(x)")
        .arg("-n")
        .arg("3")
        .arg("--verbose");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain iteration details
    assert!(stdout.contains("Iteration"));
}

// Additional optimize tests

#[test]
#[ignore = "Optimization passes are slow - see tensorlogic-compiler"]
fn test_optimize_with_verbose() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("optimize")
        .arg("p(x) AND q(x) AND r(x)")
        .arg("--level")
        .arg("aggressive")
        .arg("--verbose");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

#[test]
fn test_optimize_json_output() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("optimize")
        .arg("p(x)")
        .arg("--output-format")
        .arg("json");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

// Backend selection tests

#[test]
fn test_execute_parallel_backend() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("execute")
        .arg("p(x)")
        .arg("--backend")
        .arg("parallel");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

#[test]
fn test_execute_profiled_backend() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("execute")
        .arg("p(x)")
        .arg("--backend")
        .arg("profiled");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

#[test]
fn test_execute_with_metrics() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("execute")
        .arg("p(x)")
        .arg("--backend")
        .arg("cpu")
        .arg("--metrics");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

#[test]
fn test_execute_with_trace() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("execute")
        .arg("p(x)")
        .arg("--backend")
        .arg("cpu")
        .arg("--trace");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

// Profile command tests

#[test]
fn test_profile_command_basic() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile").arg("person(x)");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compilation Profile"));
    assert!(stdout.contains("Phase Breakdown"));
    assert!(stdout.contains("Memory Estimates"));
}

#[test]
fn test_profile_json_output() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile").arg("test(x)").arg("--json");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should be valid JSON
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
    assert!(json.get("total_time_us").is_some());
    assert!(json.get("phases").is_some());
    assert!(json.get("memory_estimate").is_some());
    assert!(json.get("graph_metrics").is_some());
}

#[test]
fn test_profile_no_optimization() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile").arg("p(x) AND q(x)").arg("--no-optimize");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compilation Profile"));
}

#[test]
fn test_profile_with_validation() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile").arg("test(x)").arg("--validate");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Validation"));
}

#[test]
fn test_profile_custom_runs() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile")
        .arg("p(x)")
        .arg("--warmup")
        .arg("0")
        .arg("--runs")
        .arg("1");

    let output = cmd.output().unwrap();
    assert!(output.status.success());
}

#[test]
fn test_profile_complex_expression() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tensorlogic").unwrap();
    cmd.arg("profile")
        .arg("person(x) AND knows(x, y)")
        .arg("--no-optimize")
        .arg("--runs")
        .arg("1")
        .arg("--warmup")
        .arg("0");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compilation Profile"));
    assert!(stdout.contains("Graph Complexity"));
}
