//! Integration tests for tensorlogic-cli

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn get_cli_binary() -> PathBuf {
    // Get the path to the compiled binary
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test executable name
    path.pop(); // Remove 'deps'

    // Try debug build first, then release
    let debug_path = path.join("tensorlogic");
    if debug_path.exists() {
        debug_path
    } else {
        path.pop(); // Remove 'debug'
        path.join("release").join("tensorlogic")
    }
}

fn run_cli(args: &[&str]) -> (String, String, i32) {
    let binary = get_cli_binary();
    let output = Command::new(&binary)
        .args(args)
        .output()
        .expect("Failed to execute CLI");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);

    (stdout, stderr, code)
}

#[test]
fn test_basic_compilation() {
    let (stdout, stderr, code) = run_cli(&["knows(x, y)", "--quiet"]);
    if code != 0 {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    assert_eq!(code, 0, "Compilation should succeed");
    assert!(!stdout.is_empty(), "Should produce output");
}

#[test]
fn test_output_format_json() {
    let (stdout, _stderr, code) = run_cli(&["knows(x, y)", "--output-format", "json", "--quiet"]);
    assert_eq!(code, 0, "Compilation should succeed");

    // Verify it's valid JSON
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");
}

#[test]
fn test_output_format_dot() {
    let (stdout, _stderr, code) = run_cli(&["knows(x, y)", "--output-format", "dot", "--quiet"]);
    assert_eq!(code, 0, "Compilation should succeed");
    assert!(
        stdout.contains("digraph"),
        "DOT output should contain digraph"
    );
}

#[test]
fn test_output_format_stats() {
    let (stdout, _stderr, code) = run_cli(&["knows(x, y)", "--output-format", "stats", "--quiet"]);
    assert_eq!(code, 0, "Compilation should succeed");
    assert!(
        stdout.contains("Tensors:"),
        "Stats should contain tensor count"
    );
    assert!(stdout.contains("Nodes:"), "Stats should contain node count");
}

#[test]
fn test_analyze_flag() {
    let (stdout, _stderr, code) = run_cli(&["knows(x, y) AND likes(y, z)", "--analyze", "--quiet"]);
    assert_eq!(code, 0, "Compilation should succeed");
    assert!(
        stdout.contains("FLOPs:") || stdout.contains("Estimated Complexity:"),
        "Analysis should show complexity metrics"
    );
}

#[test]
fn test_domain_definition() {
    let (stdout, _stderr, code) = run_cli(&[
        "knows(x, y)",
        "--domains",
        "Person:100",
        "--quiet",
        "--output-format",
        "stats",
    ]);
    assert_eq!(code, 0, "Compilation with domain should succeed");
    assert!(!stdout.is_empty());
}

#[test]
fn test_validation_success() {
    let (stdout, stderr, code) = run_cli(&["knows(x, y) AND likes(y, z)", "--validate", "--quiet"]);
    if code != 0 {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    assert_eq!(code, 0, "Valid expression should pass validation");
    assert!(!stdout.is_empty());
}

#[test]
fn test_strategy_soft_differentiable() {
    let (stdout, _stderr, code) =
        run_cli(&["p AND q", "--strategy", "soft_differentiable", "--quiet"]);
    assert_eq!(code, 0, "Soft differentiable strategy should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_strategy_hard_boolean() {
    let (stdout, _stderr, code) = run_cli(&["p AND q", "--strategy", "hard_boolean", "--quiet"]);
    assert_eq!(code, 0, "Hard boolean strategy should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_strategy_fuzzy_godel() {
    let (stdout, _stderr, code) = run_cli(&["p AND q", "--strategy", "fuzzy_godel", "--quiet"]);
    assert_eq!(code, 0, "Fuzzy Gödel strategy should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_complex_expression() {
    let (stdout, _stderr, code) =
        run_cli(&["(knows(x, y) AND likes(y, z)) OR friends(x, z)", "--quiet"]);
    assert_eq!(code, 0, "Complex expression should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_exists_quantifier() {
    let (stdout, stderr, code) = run_cli(&[
        "EXISTS x IN Person. knows(x, alice)",
        "--domains",
        "Person:100",
        "--quiet",
    ]);
    if code != 0 {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    assert_eq!(code, 0, "EXISTS quantifier should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_forall_quantifier() {
    let (stdout, _stderr, code) = run_cli(&[
        "FORALL x IN Person. mortal(x)",
        "--domains",
        "Person:100",
        "--quiet",
    ]);
    assert_eq!(code, 0, "FORALL quantifier should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_help_command() {
    let (stdout, _stderr, code) = run_cli(&["--help"]);
    assert_eq!(code, 0, "Help should succeed");
    assert!(
        stdout.contains("tensorlogic"),
        "Help should mention program name"
    );
    assert!(
        stdout.contains("USAGE") || stdout.contains("Usage"),
        "Help should show usage"
    );
}

#[test]
fn test_version_command() {
    let (stdout, _stderr, code) = run_cli(&["--version"]);
    assert_eq!(code, 0, "Version should succeed");
    assert!(
        stdout.contains("tensorlogic") || stdout.contains("0.1.0"),
        "Version should show version number"
    );
}

#[test]
fn test_no_color_flag() {
    let (stdout, _stderr, code) = run_cli(&["knows(x, y)", "--no-color", "--quiet"]);
    assert_eq!(code, 0, "No color flag should work");
    assert!(!stdout.is_empty());
}

#[test]
fn test_debug_mode() {
    let (_stdout, stderr, code) = run_cli(&["knows(x, y)", "--debug", "--quiet"]);
    assert_eq!(code, 0, "Debug mode should work");
    assert!(
        stderr.contains("Parsed expression") || stderr.contains("Context"),
        "Debug mode should show debug info"
    );
}

#[test]
fn test_file_output() {
    let temp_dir = std::env::temp_dir();
    let output_file = temp_dir.join("test_output.json");

    // Clean up if exists
    let _ = fs::remove_file(&output_file);

    let (_stdout, _stderr, code) = run_cli(&[
        "knows(x, y)",
        "--output",
        output_file.to_str().unwrap(),
        "--output-format",
        "json",
        "--quiet",
    ]);

    assert_eq!(code, 0, "File output should succeed");
    assert!(output_file.exists(), "Output file should be created");

    // Verify content
    let content = fs::read_to_string(&output_file).unwrap();
    let _: serde_json::Value =
        serde_json::from_str(&content).expect("Output file should contain valid JSON");

    // Clean up
    let _ = fs::remove_file(&output_file);
}

#[test]
fn test_completion_bash() {
    let (stdout, _stderr, code) = run_cli(&["completion", "bash"]);
    assert_eq!(code, 0, "Bash completion generation should succeed");
    assert!(
        stdout.contains("tensorlogic"),
        "Completion should reference CLI name"
    );
}

#[test]
fn test_completion_zsh() {
    let (stdout, _stderr, code) = run_cli(&["completion", "zsh"]);
    assert_eq!(code, 0, "Zsh completion generation should succeed");
    assert!(
        stdout.contains("tensorlogic") || stdout.contains("compdef"),
        "Completion should be valid zsh"
    );
}

#[test]
fn test_completion_fish() {
    let (stdout, _stderr, code) = run_cli(&["completion", "fish"]);
    assert_eq!(code, 0, "Fish completion generation should succeed");
    assert!(
        stdout.contains("tensorlogic") || stdout.contains("complete"),
        "Completion should be valid fish"
    );
}

#[test]
fn test_config_show() {
    let (stdout, _stderr, code) = run_cli(&["config", "show"]);
    // Config show may succeed with default config even if no file exists
    assert!(code == 0 || code == 1, "Config show should run");
    // If successful, should show TOML-like output
    if code == 0 {
        assert!(stdout.contains("strategy") || stdout.contains("=") || stdout.is_empty());
    }
}

#[test]
fn test_config_path() {
    let (stdout, _stderr, code) = run_cli(&["config", "path"]);
    assert_eq!(code, 0, "Config path should succeed");
    assert!(
        stdout.contains(".tensorlogicrc") || stdout.contains("tensorlogic"),
        "Should show config path"
    );
}

#[test]
fn test_convert_json_to_yaml() {
    let json_expr = r#"{"Pred":{"name":"test","args":[{"Var":"x"}]}}"#;

    let (stdout, _stderr, code) =
        run_cli(&["convert", json_expr, "--from", "json", "--to", "yaml"]);

    assert_eq!(code, 0, "JSON to YAML conversion should succeed");
    assert!(
        stdout.contains("Pred") || stdout.contains("name"),
        "YAML output should contain expression"
    );
}

#[test]
fn test_convert_expr_to_json() {
    let (stdout, _stderr, code) =
        run_cli(&["convert", "knows(x, y)", "--from", "expr", "--to", "json"]);

    assert_eq!(code, 0, "Expression to JSON conversion should succeed");
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");
}

#[test]
fn test_convert_expr_to_yaml() {
    let (stdout, _stderr, code) =
        run_cli(&["convert", "knows(x, y)", "--from", "expr", "--to", "yaml"]);

    assert_eq!(code, 0, "Expression to YAML conversion should succeed");
    assert!(
        stdout.contains("Pred") || stdout.contains("knows"),
        "YAML should contain expression"
    );
}

#[test]
fn test_convert_pretty_print() {
    let (stdout, _stderr, code) = run_cli(&[
        "convert", "p AND q", "--from", "expr", "--to", "expr", "--pretty",
    ]);

    assert_eq!(code, 0, "Pretty print should succeed");
    assert!(
        stdout.contains("AND") && (stdout.contains("p") || stdout.contains("q")),
        "Pretty printed expression should contain operators"
    );
}

#[test]
fn test_arithmetic_expression() {
    let (stdout, _stderr, code) = run_cli(&["age(x) + 10", "--quiet"]);
    assert_eq!(code, 0, "Arithmetic expression should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_comparison_expression() {
    let (stdout, _stderr, code) = run_cli(&["age(x) > 18", "--quiet"]);
    assert_eq!(code, 0, "Comparison expression should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_conditional_expression() {
    let (stdout, _stderr, code) =
        run_cli(&["IF age(x) >= 18 THEN adult(x) ELSE child(x)", "--quiet"]);
    assert_eq!(code, 0, "Conditional expression should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_invalid_strategy() {
    let (_stdout, _stderr, code) =
        run_cli(&["knows(x, y)", "--strategy", "invalid_strategy", "--quiet"]);
    assert_ne!(code, 0, "Invalid strategy should fail");
}

#[test]
fn test_multiple_domains() {
    let (stdout, _stderr, code) = run_cli(&[
        "knows(x, y)",
        "--domains",
        "Person:100",
        "--domains",
        "City:50",
        "--quiet",
    ]);
    assert_eq!(code, 0, "Multiple domains should work");
    assert!(!stdout.is_empty());
}
